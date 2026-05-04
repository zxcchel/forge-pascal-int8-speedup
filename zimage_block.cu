/*
 * zimage_block.cu — v8
 *
 * Новое в v8:
 *  - INT8 GEMM kernel через __dp4a (sm_61+ GP102 die).
 *    Weights:    int8 [N, K] + fp32 scale_w [N]   (per-channel, символическая)
 *    Activations: int8 [M, K] + fp32 scale_x [M]  (per-token, dynamic)
 *    Output:     fp32 [M, N] + optional bias
 *  - Quantize-on-the-fly kernel: fp16/bf16/fp32 [M, K] -> int8 + scale_x.
 *
 *  v6 FP32 GEMM (autotuned BK=8, TM=4, TN=16) сохранён как fallback
 *  для слоёв, где INT8 квантизация ещё не сделана.
 *
 *  Архитектурные решения по INT8 GEMM:
 *    - BK=32 (4 int8 элемента на DP4A инструкцию * 8 итераций)
 *    - BM=BN=128, TM=TN=8, 256 потоков/блок
 *    - INT32 accumulator, dequant в fp32 в эпилоге через scale_x[m] * scale_w[n]
 *    - Smem int8 [128][33] * 2 buffers ≈ 8.4KB на x и на w → 16.9KB total
 *    - Tx-stride=1 layout (как в v7) — bank-conflict-free
 *    - Acc safety: для K ≤ 130k не переполнится при насыщении ±127
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

 // ---------------------------------------------------------------------------
 // Helpers
 // ---------------------------------------------------------------------------

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t err = (x);                                                 \
        if (err != cudaSuccess) {                                              \
            printf("CUDA error at %s:%d — %s\n", __FILE__, __LINE__,          \
                   cudaGetErrorString(err));                                   \
        }                                                                      \
    } while (0)

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int mask = 16; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
}

__device__ __forceinline__ float fp8_to_f32_v5(uint8_t b) {
    uint32_t s = (uint32_t)(b >> 7) << 31;
    uint32_t e = (b >> 3) & 0xF;
    uint32_t m = b & 0x7;
    uint32_t f = s | (e ? ((e + 120u) << 23 | m << 20) : (m << 20));
    return __uint_as_float(f);
}

__device__ __forceinline__ float load_x_to_f32(const float* p) {
    return __ldg(p);
}
__device__ __forceinline__ float load_x_to_f32(const __half* p) {
    return __half2float(__ldg(p));
}
__device__ __forceinline__ float load_x_to_f32(const unsigned short* p) {
    return __uint_as_float((unsigned)__ldg(p) << 16);
}

// ===========================================================================
// Kernel 1: Fused RMSNorm
// ===========================================================================

template <int BLOCK_THREADS>
__global__ void rms_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int N, float inv_N, float eps)
{
    int row = blockIdx.x;
    const float* x_row = x + row * N;
    float* out_row = out + row * N;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < N; i += BLOCK_THREADS) {
        float v = x_row[i];
        sum_sq += v * v;
    }

    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float smem[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) smem[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < (BLOCK_THREADS / 32)) ? smem[lane_id] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
    }
    __syncthreads();

    __shared__ float rstd_shared;
    if (threadIdx.x == 0) {
        rstd_shared = rsqrtf(sum_sq * inv_N + eps);
    }
    __syncthreads();
    float rstd = rstd_shared;

    for (int i = threadIdx.x; i < N; i += BLOCK_THREADS) {
        out_row[i] = x_row[i] * rstd * weight[i];
    }
}

// ===========================================================================
// Kernel 2: fp8 Dequant + GEMM  v6 autotuned (FP32 fallback)
// ===========================================================================

#define V6_BM  128
#define V6_BN  128
#define V6_BK  8
#define V6_TM  4
#define V6_TN  16
#define V6_TX  8
#define V6_TY  32

template <typename XType>
__global__ void fp8_gemm_v6_kernel(
    const XType* __restrict__ x,
    const uint8_t* __restrict__ w,
    float* __restrict__ out,
    float scale_w,
    int M, int N, int K)
{
    __shared__ float smem_x[2][V6_BM][V6_BK + 1];
    __shared__ float smem_w[2][V6_BN][V6_BK + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * V6_TX + tx;

    int blk_m = blockIdx.y * V6_BM;
    int blk_n = blockIdx.x * V6_BN;

    float acc[V6_TM][V6_TN];
#pragma unroll
    for (int i = 0; i < V6_TM; i++)
#pragma unroll
        for (int j = 0; j < V6_TN; j++)
            acc[i][j] = 0.0f;

    int num_tiles = (K + V6_BK - 1) / V6_BK;

#pragma unroll
    for (int i = 0; i < 4; i++) {
        int e = tid + i * 256;
        int mi = e / V6_BK;
        int ki = e % V6_BK;
        int gm = blk_m + mi, gk = ki;
        smem_x[0][mi][ki] = (gm < M && gk < K) ? load_x_to_f32(&x[gm * K + gk]) : 0.0f;
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int e = tid + i * 256;
        int ni = e / V6_BK;
        int ki = e % V6_BK;
        int gn = blk_n + ni, gk = ki;
        uint8_t raw = (gn < N && gk < K) ? __ldg(&w[gn * K + gk]) : 0;
        smem_w[0][ni][ki] = fp8_to_f32_v5(raw) * scale_w;
    }
    __syncthreads();

    float prefetch_x[4];
    float prefetch_w[4];

    for (int kt = 0; kt < num_tiles; kt++) {
        int buf = kt & 1;
        int nxt = 1 - buf;
        int next_tile = kt + 1;
        bool has_next = (next_tile < num_tiles);

        if (has_next) {
            int k0 = next_tile * V6_BK;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                int e = tid + i * 256;
                int mi = e / V6_BK, ki = e % V6_BK;
                int gm = blk_m + mi, gk = k0 + ki;
                prefetch_x[i] = (gm < M && gk < K)
                    ? load_x_to_f32(&x[gm * K + gk]) : 0.0f;
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
                int e = tid + i * 256;
                int ni = e / V6_BK, ki = e % V6_BK;
                int gn = blk_n + ni, gk = k0 + ki;
                uint8_t raw = (gn < N && gk < K) ? __ldg(&w[gn * K + gk]) : 0;
                prefetch_w[i] = fp8_to_f32_v5(raw) * scale_w;
            }
        }

#pragma unroll
        for (int k = 0; k < V6_BK; k++) {
            float x_reg[V6_TM];
#pragma unroll
            for (int i = 0; i < V6_TM; i++)
                x_reg[i] = smem_x[buf][ty * V6_TM + i][k];

            float w_reg[V6_TN];
#pragma unroll
            for (int j = 0; j < V6_TN; j++)
                w_reg[j] = smem_w[buf][tx + j * V6_TX][k];

#pragma unroll
            for (int i = 0; i < V6_TM; i++)
#pragma unroll
                for (int j = 0; j < V6_TN; j++)
                    acc[i][j] += x_reg[i] * w_reg[j];
        }

        if (has_next) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                int e = tid + i * 256;
                int mi = e / V6_BK, ki = e % V6_BK;
                smem_x[nxt][mi][ki] = prefetch_x[i];
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
                int e = tid + i * 256;
                int ni = e / V6_BK, ki = e % V6_BK;
                smem_w[nxt][ni][ki] = prefetch_w[i];
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < V6_TM; i++) {
        int m = blk_m + ty * V6_TM + i;
        if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < V6_TN; j++) {
            int n = blk_n + tx + j * V6_TX;
            if (n < N) out[m * N + n] = acc[i][j];
        }
    }
}

// ===========================================================================
// Kernel 2b: Quantize activations to INT8 (per-token absmax)
//
// Один thread block обрабатывает одну row. 256 потоков, кооперативный absmax
// и квантизация.
// ===========================================================================

template <typename XType>
__global__ void quantize_x_int8_kernel(
    const XType* __restrict__ x,
    int8_t* __restrict__ x_q,
    float* __restrict__ scale_x,
    int M, int K)
{
    int row = blockIdx.x;
    if (row >= M) return;

    const XType* x_row = x + row * K;
    int8_t* xq_row = x_q + row * K;

    int tid = threadIdx.x;
    const int BT = 256;

    // ── Pass 1: per-row absmax ─────────────────────────────────────────
    float my_max = 0.0f;
    for (int i = tid; i < K; i += BT) {
        float v = load_x_to_f32(&x_row[i]);
        float a = fabsf(v);
        if (a > my_max) my_max = a;
    }

    my_max = warp_reduce_max(my_max);

    __shared__ float smem_max[8];   // 256 threads / 32 = 8 warps
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    if (lane_id == 0) smem_max[warp_id] = my_max;
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane_id < 8) ? smem_max[lane_id] : 0.0f;
        v = warp_reduce_max(v);
        if (lane_id == 0) {
            float s = v / 127.0f;
            if (s < 1e-12f) s = 1e-12f;  // защита от нулевой row
            smem_max[0] = s;
            scale_x[row] = s;
        }
    }
    __syncthreads();

    float inv_scale = 1.0f / smem_max[0];

    // ── Pass 2: quantize ────────────────────────────────────────────────
    for (int i = tid; i < K; i += BT) {
        float v = load_x_to_f32(&x_row[i]);
        int q = __float2int_rn(v * inv_scale);
        if (q > 127)  q = 127;
        if (q < -127) q = -127;
        xq_row[i] = (int8_t)q;
    }
}

// ===========================================================================
// Kernel 2d: INT8 GEMM via DP4A (v2 — 2 blocks/SM)
//
// Изменения относительно v2c:
//   * smem_w: [2][...] → [1][...], -4352 байт smem
//   * Итого smem: 26112 → 21760 байт → 2 блока/SM на GP102
//   * smem_x по-прежнему двойной (бо́льший, скрывает latency)
//   * prefetch_w в регистрах → commit в smem_w[0] (единственный буфер)
//   * Всё остальное: без изменений
//
// Smem layout:
//   smem_x [2][BM=128][KDWORDS+1=17] × 4 байт = 17408 байт
//   smem_w [1][BN= 64][KDWORDS+1=17] × 4 байт =  4352 байт
//   Итого                                        21760 байт  (<24576 ✓)
// ===========================================================================

#define I8_BM  128
#define I8_BN  64
#define I8_BK  64
#define I8_TM  8
#define I8_TN  4

#define I8_TX  (I8_BN / I8_TN)            // 16
#define I8_TY  (I8_BM / I8_TM)            // 16
#define THREADS_PER_BLOCK (I8_TX * I8_TY) // 256

#define KDWORDS (I8_BK / 4)               // 16 (4 × int8 → 1 × int32)
#define LOADS_X ((I8_BM * KDWORDS) / THREADS_PER_BLOCK) // 8
#define LOADS_W ((I8_BN * KDWORDS) / THREADS_PER_BLOCK) // 4
__device__ __forceinline__ float cvt_out(float v, float) { return v; }
__device__ __forceinline__ half cvt_out(float v, half) { return __float2half(v); }
template <typename OutT>
__launch_bounds__(THREADS_PER_BLOCK, 2)
__global__ void int8_gemm_dp4a_kernel(
    const int8_t* __restrict__ x,
    const int8_t* __restrict__ w,
    const float* __restrict__ scale_x,
    const float* __restrict__ scale_w,
    const float* __restrict__ bias,
    OutT* __restrict__ out,      // <--- ТЕПЕРЬ ТУТ ШАБЛОННЫЙ ТИП
    int M, int N, int K)
{
    // smem_x — двойной буфер (pipeline: BM×BK активации)
    // smem_w — одинарный (prefetch идёт через регистры prefetch_w[])
    // Padding +1 dword за строку — bank-conflict-free для 32-банков × 4 байт
    __shared__ int32_t smem_x[2][I8_BM][KDWORDS + 1]; // 17408 байт
    __shared__ int32_t smem_w[1][I8_BN][KDWORDS + 1]; //  4352 байт

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * I8_TX + tx;

    const int blk_m = blockIdx.y * I8_BM;
    const int blk_n = blockIdx.x * I8_BN;

    // Регистровые аккумуляторы
    int32_t acc[I8_TM][I8_TN];
#pragma unroll
    for (int i = 0; i < I8_TM; i++)
#pragma unroll
        for (int j = 0; j < I8_TN; j++)
            acc[i][j] = 0;

    const int num_tiles = (K + I8_BK - 1) / I8_BK;

    // ── Bootstrap: загружаем tile 0 ───────────────────────────────────────

    // smem_x[0]
#pragma unroll
    for (int i = 0; i < LOADS_X; i++) {
        const int e = tid + i * THREADS_PER_BLOCK;
        const int mi = e / KDWORDS;
        const int ki = e % KDWORDS;
        const int gm = blk_m + mi;
        const int gk = ki * 4;
        int32_t v = 0;
        if (gm < M && gk < K)
            v = __ldg(reinterpret_cast<const int32_t*>(&x[gm * K + gk]));
        smem_x[0][mi][ki] = v;
    }

    // smem_w[0]  (единственный буфер W)
#pragma unroll
    for (int i = 0; i < LOADS_W; i++) {
        const int e = tid + i * THREADS_PER_BLOCK;
        const int ni = e / KDWORDS;
        const int ki = e % KDWORDS;
        const int gn = blk_n + ni;
        const int gk = ki * 4;
        int32_t v = 0;
        if (gn < N && gk < K)
            v = __ldg(reinterpret_cast<const int32_t*>(&w[gn * K + gk]));
        smem_w[0][ni][ki] = v;
    }

    __syncthreads();

    // Регистровые prefetch-буферы
    int32_t prefetch_x[LOADS_X];
    int32_t prefetch_w[LOADS_W];

    // ── Основной цикл ─────────────────────────────────────────────────────
    for (int kt = 0; kt < num_tiles; kt++) {
        const int buf = kt & 1;          // текущий буфер X
        const int nxt = 1 - buf;         // следующий буфер X
        const int next_k0 = (kt + 1) * I8_BK;
        const bool has_next = (kt + 1 < num_tiles);

        // ── Prefetch из gmem в регистры ──────────────────────────────────
        if (has_next) {
            // Активации X → prefetch_x[]
#pragma unroll
            for (int i = 0; i < LOADS_X; i++) {
                const int e = tid + i * THREADS_PER_BLOCK;
                const int mi = e / KDWORDS;
                const int ki = e % KDWORDS;
                const int gm = blk_m + mi;
                const int gk = next_k0 + ki * 4;
                prefetch_x[i] = (gm < M && gk < K)
                    ? __ldg(reinterpret_cast<const int32_t*>(&x[gm * K + gk]))
                    : 0;
            }
            // Веса W → prefetch_w[]
#pragma unroll
            for (int i = 0; i < LOADS_W; i++) {
                const int e = tid + i * THREADS_PER_BLOCK;
                const int ni = e / KDWORDS;
                const int ki = e % KDWORDS;
                const int gn = blk_n + ni;
                const int gk = next_k0 + ki * 4;
                prefetch_w[i] = (gn < N && gk < K)
                    ? __ldg(reinterpret_cast<const int32_t*>(&w[gn * K + gk]))
                    : 0;
            }
        }

        // ── Вычисление на smem_x[buf] × smem_w[0] ────────────────────────
#pragma unroll
        for (int kk = 0; kk < KDWORDS; kk++) {

            int32_t x_reg[I8_TM];
#pragma unroll
            for (int i = 0; i < I8_TM; i++)
                x_reg[i] = smem_x[buf][ty * I8_TM + i][kk];

            int32_t w_reg[I8_TN];
#pragma unroll
            for (int j = 0; j < I8_TN; j++)
                w_reg[j] = smem_w[0][tx + j * I8_TX][kk]; // всегда [0]

#pragma unroll
            for (int i = 0; i < I8_TM; i++)
#pragma unroll
                for (int j = 0; j < I8_TN; j++)
                    acc[i][j] = __dp4a(x_reg[i], w_reg[j], acc[i][j]);
        }

        // ── Commit prefetch в smem ────────────────────────────────────────
        if (has_next) {
            // X → двойной буфер (nxt)
#pragma unroll
            for (int i = 0; i < LOADS_X; i++) {
                const int e = tid + i * THREADS_PER_BLOCK;
                const int mi = e / KDWORDS;
                const int ki = e % KDWORDS;
                smem_x[nxt][mi][ki] = prefetch_x[i];
            }
            // W → единственный буфер [0] (overwrite — compute уже завершён)
#pragma unroll
            for (int i = 0; i < LOADS_W; i++) {
                const int e = tid + i * THREADS_PER_BLOCK;
                const int ni = e / KDWORDS;
                const int ki = e % KDWORDS;
                smem_w[0][ni][ki] = prefetch_w[i]; // [0], не [nxt]
            }
        }

        __syncthreads(); // барьер перед следующей итерацией
    }

    // ── Epilogue: int32 → fp32 dequant + bias + запись ────────────────────

    float scale_x_local[I8_TM];
#pragma unroll
    for (int i = 0; i < I8_TM; i++) {
        const int m = blk_m + ty * I8_TM + i;
        scale_x_local[i] = (m < M) ? __ldg(&scale_x[m]) : 0.0f;
    }

    float scale_w_local[I8_TN];
    float bias_local[I8_TN];
#pragma unroll
    for (int j = 0; j < I8_TN; j++) {
        const int n = blk_n + tx + j * I8_TX;
        scale_w_local[j] = (n < N) ? __ldg(&scale_w[n]) : 0.0f;
        bias_local[j] = (n < N && bias != nullptr) ? __ldg(&bias[n]) : 0.0f;
    }

#pragma unroll
    for (int i = 0; i < I8_TM; i++) {
        const int m = blk_m + ty * I8_TM + i;
        if (m >= M) continue;
        const float sx = scale_x_local[i];
#pragma unroll
        for (int j = 0; j < I8_TN; j++) {
            const int n = blk_n + tx + j * I8_TX;
            if (n < N) {
                // Считаем финальное значение во float
                float f_val = (float)acc[i][j] * sx * scale_w_local[j] + bias_local[j];
                // Безопасно кастим в OutT (float или half) и пишем в память
                out[m * N + n] = cvt_out(f_val, OutT());
            }
        }
    }
}

// ===========================================================================
// Kernel 3: Apply RoPE in-place
// ===========================================================================

__global__ void apply_rope_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ freqs,
    int B, int seq, int nh, int nkh, int hd)
{
    int idx = blockIdx.x;
    int pair = blockIdx.y * blockDim.x + threadIdx.x;
    if (pair >= hd / 2) return;

    int b = idx / (seq * nh);
    int s = (idx / nh) % seq;
    int h = idx % nh;

    float cos_val = freqs[s * hd + pair * 2];
    float sin_val = freqs[s * hd + pair * 2 + 1];

    if (h < nh) {
        int base = ((b * seq + s) * nh + h) * hd + pair * 2;
        float q0 = q[base], q1 = q[base + 1];
        q[base] = q0 * cos_val - q1 * sin_val;
        q[base + 1] = q0 * sin_val + q1 * cos_val;
    }
    if (h < nkh) {
        int base = ((b * seq + s) * nkh + h) * hd + pair * 2;
        float k0 = k[base], k1 = k[base + 1];
        k[base] = k0 * cos_val - k1 * sin_val;
        k[base + 1] = k0 * sin_val + k1 * cos_val;
    }
}

// ===========================================================================
// Kernel 4c: Flash Attention forward (sm61, v3 — warp-per-row)
//
// Проблема v2:
//   Один поток на строку Q → q_reg[HD] + acc[HD] на поток.
//   HD=128: 288 регистров нужно, 255 максимум → неизбежный спилл.
//   HD=64:  192 + overhead → тоже спилл.
//
// Решение (v3): один ВАРП на строку Q.
//   Каждый поток варпа держит HD/32 элементов.
//   HD=128 → 4 float/поток, HD=64 → 2 float/поток.
//   QK dot product = warp-reduce через __shfl_xor_sync.
//   m_i, l_i, scores[], p[] — одинаковы для всех lane в варпе.
//
// Параметры:
//   BM_ROWS = строк Q на блок = кол-во варпов в блоке (4)
//   blockDim.x = BM_ROWS * 32 = 128 потоков
//   BN = тайл по N (16 для HD=128, 32 для HD=64)
//
// Smem:
//   HD=128, BN=16: 2×16×128×4 = 16 384 байт → 3 блока/SM ✓
//   HD=64,  BN=32: 2×32×64×4  = 16 384 байт → 3 блока/SM ✓
//
// Регистры на поток (оценка):
//   HD=128: q_reg[4]+acc[4]+scores[16]+p[16] ≈ 60 reg → ~7 блоков/SM (smem лимит)
//   HD=64:  q_reg[2]+acc[2]+scores[32]+p[32] ≈ 80 reg → ~6 блоков/SM (smem лимит)
//
// Запуск:
//   dim3 grid(B*nh, (M + BM_ROWS - 1) / BM_ROWS);
//   dim3 block(BM_ROWS * 32);   // = 128
//   flash_attn_fwd_v3_kernel<64,  4, 32><<<grid, block, 0, stream>>>(...)
//   flash_attn_fwd_v3_kernel<128, 4, 16><<<grid, block, 0, stream>>>(...)
// ===========================================================================

#define FA2_BM   32
#define FA2_BN   16
#define FA2_TX    8
#define FA2_TY    8
#define FA2_TM   (FA2_BM / FA2_TY)   // = 4 строк Q на поток

template<int HD>
__launch_bounds__(FA2_TX* FA2_TY, 1)
__global__ void flash_attn_fwd_v4_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    float softmax_scale,
    int B, int nh, int M, int N,
    int stride_qb, int stride_qh, int stride_qm,
    int stride_kb, int stride_kh, int stride_kn,
    int stride_vb, int stride_vh, int stride_vn,
    int stride_ob, int stride_oh, int stride_om)
{
    static_assert(HD % FA2_TX == 0, "HD must be divisible by FA2_TX");
    constexpr int HD_PER_TX = HD / FA2_TX;     // 16 при HD=128, 8 при HD=64

    const int tx = threadIdx.x & (FA2_TX - 1);  // 0..7  → d-направление
    const int ty = threadIdx.x / FA2_TX;        // 0..7  → m-направление
    const int tid = threadIdx.x;

    const int bh = blockIdx.y;
    const int b = bh / nh;
    const int h = bh % nh;
    const int blk_m = blockIdx.x * FA2_BM;

    // ── Shared memory ──────────────────────────────────────────────────────
    __shared__ float smem_q[FA2_BM][HD + 1];
    __shared__ float smem_k[FA2_BN][HD + 1];
    __shared__ float smem_v[FA2_BN][HD + 1];
    __shared__ float smem_s[FA2_BM][FA2_BN];   // scratch для QK reduce

    const float* Q_base = Q + b * stride_qb + h * stride_qh;
    const float* K_base = K + b * stride_kb + h * stride_kh;
    const float* V_base = V + b * stride_vb + h * stride_vh;

    // ── Загрузка Q-тайла в smem (один раз на весь kernel) ──────────────────
    constexpr int Q_ELEMS = FA2_BM * HD;
    constexpr int Q_PER_THREAD = Q_ELEMS / (FA2_TX * FA2_TY);  // 64
#pragma unroll
    for (int i = 0; i < Q_PER_THREAD; i++) {
        const int e = tid + i * (FA2_TX * FA2_TY);
        const int mi = e / HD;
        const int di = e % HD;
        const int gm = blk_m + mi;
        smem_q[mi][di] = (gm < M)
            ? __ldg(Q_base + gm * stride_qm + di)
            : 0.0f;
    }

    // ── Регистровые состояния ──────────────────────────────────────────────
    // Поток (tx, ty) отвечает за:
    //   - строки Q: ty*TM .. ty*TM+TM-1
    //   - d-элементы acc_o: tx*HD_PER_TX .. tx*HD_PER_TX+HD_PER_TX-1
    float acc_o[FA2_TM][HD_PER_TX];
    float m_i[FA2_TM];
    float l_i[FA2_TM];
#pragma unroll
    for (int i = 0; i < FA2_TM; i++) {
        m_i[i] = -1e30f;
        l_i[i] = 0.0f;
#pragma unroll
        for (int d = 0; d < HD_PER_TX; d++)
            acc_o[i][d] = 0.0f;
    }

    __syncthreads();

    // ═════════════════════════════════════════════════════════════════════
    // Главный цикл по N-тайлам
    // ═════════════════════════════════════════════════════════════════════
    for (int n_start = 0; n_start < N; n_start += FA2_BN) {

        // ── Загрузка K и V в smem (16 × 128 = 2048 / 64 = 32 эл/тред) ──────
        constexpr int KV_ELEMS = FA2_BN * HD;
        constexpr int KV_PER_THREAD = KV_ELEMS / (FA2_TX * FA2_TY);
#pragma unroll
        for (int i = 0; i < KV_PER_THREAD; i++) {
            const int e = tid + i * (FA2_TX * FA2_TY);
            const int ni = e / HD;
            const int di = e % HD;
            const int gn = n_start + ni;
            const bool valid = (gn < N);
            smem_k[ni][di] = valid ? __ldg(K_base + gn * stride_kn + di) : 0.0f;
            smem_v[ni][di] = valid ? __ldg(V_base + gn * stride_vn + di) : 0.0f;
        }
        __syncthreads();

        // ── QK: каждый поток считает partial dot для своих TM строк × всех BN столбцов
        // Поток (tx, ty) суммирует d-куски [tx*HD_PER_TX .. +HD_PER_TX-1].
        // Полный dot по HD получается после reduce по tx (через smem).
        float s_partial[FA2_TM][FA2_BN];
#pragma unroll
        for (int i = 0; i < FA2_TM; i++)
#pragma unroll
            for (int j = 0; j < FA2_BN; j++)
                s_partial[i][j] = 0.0f;

        const int d_base = tx * HD_PER_TX;
#pragma unroll
        for (int dk = 0; dk < HD_PER_TX; dk++) {
            const int d = d_base + dk;

            // Загрузить столбец Q (TM значений) и столбец K (BN значений)
            float q_reg[FA2_TM];
#pragma unroll
            for (int i = 0; i < FA2_TM; i++)
                q_reg[i] = smem_q[ty * FA2_TM + i][d];

            float k_reg[FA2_BN];
#pragma unroll
            for (int j = 0; j < FA2_BN; j++)
                k_reg[j] = smem_k[j][d];

#pragma unroll
            for (int i = 0; i < FA2_TM; i++)
#pragma unroll
                for (int j = 0; j < FA2_BN; j++)
                    s_partial[i][j] += q_reg[i] * k_reg[j];
        }

        // ── Cross-tx reduce через smem ────────────────────────────────────
        // Стратегия: каждый tx пишет свой partial в slot [row][col*TX + tx],
        // потом tx==0 суммирует TX значений и пишет результат в [row][col].
        //
        // Не лезет: smem_s имеет размер [BM][BN] = [32][16] = 2 KB.
        // Нам нужно [BM][BN][TX] = 16 KB временно. Перебор.
        //
        // Альтернатива: tree-reduce через несколько проходов с __syncthreads.
        // Проще: круговая стратегия — каждый tx по очереди дописывает.
        //
        // Самый простой работающий вариант: atomicAdd на smem_s.
        // На Pascal smem atomic быстрый. 32*16*8 = 4096 atomics на блок,
        // делится по итерациям BN-loop — приемлемо.

        // Инициализация smem_s
        if (tx == 0) {
#pragma unroll
            for (int i = 0; i < FA2_TM; i++)
#pragma unroll
                for (int j = 0; j < FA2_BN; j++)
                    smem_s[ty * FA2_TM + i][j] = 0.0f;
        }
        __syncthreads();

        // Atomic-add partial values в smem_s
#pragma unroll
        for (int i = 0; i < FA2_TM; i++)
#pragma unroll
            for (int j = 0; j < FA2_BN; j++)
                atomicAdd(&smem_s[ty * FA2_TM + i][j], s_partial[i][j]);

        __syncthreads();

        // ── Online softmax: каждый поток обрабатывает свои TM строк ───────
        // Все потоки с одинаковым ty делают ОДИНАКОВУЮ работу — это OK,
        // потому что результаты m_i, l_i, p[][] идут в acc_o, который
        // у каждого потока свой (по tx).
        float p[FA2_TM][FA2_BN];
#pragma unroll
        for (int i = 0; i < FA2_TM; i++) {
            const int row = ty * FA2_TM + i;

            // Применяем scale + маску
            float s_row[FA2_BN];
            float mx = -1e30f;
#pragma unroll
            for (int j = 0; j < FA2_BN; j++) {
                const int gn = n_start + j;
                float v = (gn < N) ? smem_s[row][j] * softmax_scale : -1e30f;
                s_row[j] = v;
                mx = fmaxf(mx, v);
            }

            const float m_new = fmaxf(m_i[i], mx);
            const float alpha = __expf(m_i[i] - m_new);
            float l_acc = alpha * l_i[i];

#pragma unroll
            for (int j = 0; j < FA2_BN; j++) {
                p[i][j] = __expf(s_row[j] - m_new);
                l_acc += p[i][j];
            }

            // Rescale acc_o
#pragma unroll
            for (int d = 0; d < HD_PER_TX; d++)
                acc_o[i][d] *= alpha;

            m_i[i] = m_new;
            l_i[i] = l_acc;
        }

        // ── AV: acc_o += p · V ────────────────────────────────────────────
        // Поток считает свои HD_PER_TX d-элементов напрямую — без редьюса.
#pragma unroll
        for (int i = 0; i < FA2_TM; i++) {
#pragma unroll
            for (int d = 0; d < HD_PER_TX; d++) {
                float v_acc = 0.0f;
#pragma unroll
                for (int j = 0; j < FA2_BN; j++)
                    v_acc += p[i][j] * smem_v[j][d_base + d];
                acc_o[i][d] += v_acc;
            }
        }

        __syncthreads();  // перед перезаписью smem_k, smem_v на след. итерации
    }

    // ═════════════════════════════════════════════════════════════════════
    // Epilogue: каждый поток пишет свои строки × d-элементы
    // ═════════════════════════════════════════════════════════════════════
    const int d_base_out = tx * HD_PER_TX;   // ← добавь эту строку
#pragma unroll
    for (int i = 0; i < FA2_TM; i++) {
        const int gm = blk_m + ty * FA2_TM + i;
        if (gm >= M) continue;
        const float inv_l = __frcp_rn(l_i[i]);
        float* Out_row = Out + b * stride_ob + h * stride_oh + gm * stride_om;
#pragma unroll
        for (int d = 0; d < HD_PER_TX; d++) {
            Out_row[d_base_out + d] = acc_o[i][d] * inv_l;   // ← d_base → d_base_out
        }
    }
}


// ===========================================================================
// Host launchers
// ===========================================================================

extern "C" {

    void launch_rms_norm(
        const float* x, const float* weight, float* out,
        int rows, int N, float eps,
        cudaStream_t stream)
    {
        int block = 1;
        while (block < N && block < 1024) block <<= 1;
        dim3 grid(rows);

        if (block <= 128)      rms_norm_kernel<128> << <grid, 128, 0, stream >> > (x, weight, out, N, 1.0f / N, eps);
        else if (block <= 256) rms_norm_kernel<256> << <grid, 256, 0, stream >> > (x, weight, out, N, 1.0f / N, eps);
        else if (block <= 512) rms_norm_kernel<512> << <grid, 512, 0, stream >> > (x, weight, out, N, 1.0f / N, eps);
        else                   rms_norm_kernel<1024> << <grid, 1024, 0, stream >> > (x, weight, out, N, 1.0f / N, eps);
    }

    // FP32 GEMM (legacy)
    void launch_fp8_gemm(
        const float* x, const uint8_t* w, float* out,
        float scale_w, int M, int N, int K, cudaStream_t stream)
    {
        dim3 block(V6_TX, V6_TY);
        dim3 grid((N + V6_BN - 1) / V6_BN, (M + V6_BM - 1) / V6_BM);
        fp8_gemm_v6_kernel<float> << <grid, block, 0, stream >> > (x, w, out, scale_w, M, N, K);
    }

    void launch_fp8_gemm_f16(
        const void* x, const uint8_t* w, float* out,
        float scale_w, int M, int N, int K, cudaStream_t stream)
    {
        dim3 block(V6_TX, V6_TY);
        dim3 grid((N + V6_BN - 1) / V6_BN, (M + V6_BM - 1) / V6_BM);
        fp8_gemm_v6_kernel<__half> << <grid, block, 0, stream >> > (
            reinterpret_cast<const __half*>(x), w, out, scale_w, M, N, K);
    }

    void launch_fp8_gemm_bf16(
        const void* x, const uint8_t* w, float* out,
        float scale_w, int M, int N, int K, cudaStream_t stream)
    {
        dim3 block(V6_TX, V6_TY);
        dim3 grid((N + V6_BN - 1) / V6_BN, (M + V6_BM - 1) / V6_BM);
        fp8_gemm_v6_kernel<unsigned short> << <grid, block, 0, stream >> > (
            reinterpret_cast<const unsigned short*>(x), w, out, scale_w, M, N, K);
    }

    // INT8 quantize launchers
    void launch_quantize_x_f32(const void* x, int8_t* x_q, float* scale_x,
        int M, int K, cudaStream_t stream)
    {
        quantize_x_int8_kernel<float> << <M, 256, 0, stream >> > (
            reinterpret_cast<const float*>(x), x_q, scale_x, M, K);
    }
    void launch_quantize_x_f16(const void* x, int8_t* x_q, float* scale_x,
        int M, int K, cudaStream_t stream)
    {
        quantize_x_int8_kernel<__half> << <M, 256, 0, stream >> > (
            reinterpret_cast<const __half*>(x), x_q, scale_x, M, K);
    }
    void launch_quantize_x_bf16(const void* x, int8_t* x_q, float* scale_x,
        int M, int K, cudaStream_t stream)
    {
        quantize_x_int8_kernel<unsigned short> << <M, 256, 0, stream >> > (
            reinterpret_cast<const unsigned short*>(x), x_q, scale_x, M, K);
    }

    // INT8 GEMM (FP32 Output для z-image)
    void launch_int8_gemm_f32(
        const int8_t* x_q, const int8_t* w_q,
        const float* scale_x, const float* scale_w, const float* bias,
        float* out, int M, int N, int K, cudaStream_t stream)
    {
        dim3 block(I8_TX, I8_TY);
        dim3 grid((N + I8_BN - 1) / I8_BN, (M + I8_BM - 1) / I8_BM);
        int8_gemm_dp4a_kernel<float> << <grid, block, 0, stream >> > (
            x_q, w_q, scale_x, scale_w, bias, out, M, N, K);
    }

    // INT8 GEMM (FP16 Output для Flux)
    // Используем void* out, чтобы PyTorch не ругался на конфликты заголовков
    void launch_int8_gemm_f16(
        const int8_t* x_q, const int8_t* w_q,
        const float* scale_x, const float* scale_w, const float* bias,
        void* out, int M, int N, int K, cudaStream_t stream)
    {
        dim3 block(I8_TX, I8_TY);
        dim3 grid((N + I8_BN - 1) / I8_BN, (M + I8_BM - 1) / I8_BM);
        int8_gemm_dp4a_kernel<half> << <grid, block, 0, stream >> > (
            x_q, w_q, scale_x, scale_w, bias, reinterpret_cast<half*>(out), M, N, K);
    }

    void launch_flash_attn(
        const float* Q, const float* K, const float* V, float* Out,
        float softmax_scale,
        int B, int nh, int M, int N, int hd,
        int stride_qb, int stride_qh, int stride_qm, int /*stride_qd*/,
        int stride_kb, int stride_kh, int stride_kn, int /*stride_kd*/,
        int stride_vb, int stride_vh, int stride_vn, int /*stride_vd*/,
        int stride_ob, int stride_oh, int stride_om, int /*stride_od*/,
        cudaStream_t stream)
    {
        constexpr int BM = 32;
        dim3 grid((M + BM - 1) / BM, B * nh);
        dim3 block(FA2_TX * FA2_TY);  // 64

        if (hd == 128) {
            flash_attn_fwd_v4_kernel<128> << <grid, block, 0, stream >> > (
                Q, K, V, Out, softmax_scale, B, nh, M, N,
                stride_qb, stride_qh, stride_qm,
                stride_kb, stride_kh, stride_kn,
                stride_vb, stride_vh, stride_vn,
                stride_ob, stride_oh, stride_om);
        }
        else if (hd == 64) {
            flash_attn_fwd_v4_kernel<64> << <grid, block, 0, stream >> > (
                Q, K, V, Out, softmax_scale, B, nh, M, N,
                stride_qb, stride_qh, stride_qm,
                stride_kb, stride_kh, stride_kn,
                stride_vb, stride_vh, stride_vn,
                stride_ob, stride_oh, stride_om);
        }
    }

} // extern "C"
