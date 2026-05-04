# INT8 DP4A GEMM Patch for ForgeUI Neo

> ⚠️ **This is the very first release — still in active development.**  
> Testing was only possible on a single machine. Behavior on other hardware configurations is unknown.  
> Tested on **Flux 2 Klein 9B** and **Z Image Turbo**.  
> Weight computation speedup: **~2×** vs stock on Pascal (GP102).

**Test hardware:** Intel Core i3-10100F · 32 GB DDR4 · NVIDIA Titan X Pascal GP102 (undervolted + overclocked)

---

## Screenshots

<!-- UI comparison:  -->
<!-- Profiler traces: stock fp16 vs INT8 DP4A -->
<!-- Generated image quality comparison
 -->

Generated image quality comparison:
Z-Image Turbo
<img width="2048" height="1024" alt="images" src="https://github.com/user-attachments/assets/7d1f8340-c969-43e3-8c77-cbdceafd900e" />
Flux 2 Klein 9B
<img width="2048" height="1024" alt="images2" src="https://github.com/user-attachments/assets/d5796edd-9dfd-4540-8d5e-28bc86901ae8" />
UI comparison:
Z-Image Turbo 
FP8<img width="1901" height="949" alt="zit_ui_cublas" src="https://github.com/user-attachments/assets/ede88f32-4386-43d1-83fc-cd6dbfe4f322" />
INT8<img width="1901" height="951" alt="zit_ui_int8" src="https://github.com/user-attachments/assets/c3462a48-dd6d-402e-b933-5c032c4f3c4d" />
Flux 2 Klein 9B
FP8<img width="1901" height="944" alt="flux_ui_cublas" src="https://github.com/user-attachments/assets/1f9b2579-f4c2-462c-8424-e303a87aba14" />
INT8<img width="1899" height="946" alt="flux_ui_int8" src="https://github.com/user-attachments/assets/66354832-6217-4324-b794-c9b7d4f9e3ac" />
Profiler traces :
Z-Image Turbo 
FP8<img width="1082" height="770" alt="zit_profiler_table_cublas" src="https://github.com/user-attachments/assets/e5daf2a9-5f2d-4c7b-8d23-dc9bbab5024d" />
INT8<img width="1087" height="659" alt="zit_profiler_table_int8" src="https://github.com/user-attachments/assets/c0875efc-316b-4f37-bc4f-a33ab6cd4fe0" />
CUDA Time FP8<img width="281" height="48" alt="zit_profiler_table_cublas_time" src="https://github.com/user-attachments/assets/fbaa3969-9a4e-4384-a0f2-39897b92f42d" />
CUDA Time INT8<img width="276" height="46" alt="zit_profiler_int8_time" src="https://github.com/user-attachments/assets/44ea52e1-16ee-4ee0-96c7-166395f7383e" />
Flux 2 Klein 9B
FP8<img width="1057" height="884" alt="flux_profiler_table_cublas" src="https://github.com/user-attachments/assets/4ff17fd1-3762-4171-be4f-8c776019bcb1" />
INT8<img width="1101" height="562" alt="flux_profiler_table_int8" src="https://github.com/user-attachments/assets/3dece291-4a0c-4d99-a8d0-ada1e9847684" />
CUDA Time FP8<img width="291" height="48" alt="flux_profiler_cublas_time" src="https://github.com/user-attachments/assets/3c13f8c7-37c3-47aa-9307-2504fd870930" />
CUDA Time INT8<img width="283" height="41" alt="flux_profiler_int8_time" src="https://github.com/user-attachments/assets/2cac0b15-86fa-4afb-8d28-30f325642e4b" />
---

## What This Patch Does

This patch **replaces the standard forward path** for `torch.nn.Linear` inside ForgeUI Neo when running fp8 checkpoints.

Stock path:
```
fp8 weight → dequant to fp16 → cuBLAS fp16 GEMM
```

This patch:
```
fp8 weight → INT8 per-channel (once, lazy) → INT8×INT8 DP4A GEMM → fp32/fp16 output
```

The replacement is transparent — `ForgeOperations.Linear.forward()` and `ForgeOperationsInt8.Linear.forward()` silently route through the INT8 kernel when all gate conditions are met, falling back to stock `torch.nn.functional.linear` if they are not.

---

## ⚠️ GPU Compatibility Warning

**This patch is designed and optimized for Pascal architecture (sm_61, GP102 die): GTX 1080, GTX 1080 Ti, Titan Xp.**

| Architecture | Recommendation |
|---|---|
| Pascal (GTX 1080, 1080 Ti, Titan X) | ✅ Use this patch — ~2× speedup |
| Turing (RTX 20xx) | ⚠️ Will work but likely no benefit or slower |
| Ampere / Ada / Hopper (RTX 30xx, 40xx) | ❌ **Do not use** — will be slower than stock |

<details>
<summary>Why newer GPUs should not use this patch</summary>

Starting from Volta, NVIDIA introduced **Tensor Cores** with native INT8 matrix instructions (`IMMA`). cuBLAS automatically uses them and is significantly faster than the ALU-based `__dp4a` used in this kernel. On these GPUs the stock path (cuBLAS) is already optimal. Using this patch on Volta+ routes INT8 through regular CUDA cores, bypassing Tensor Cores entirely — a performance regression.

If you are on Ampere or newer, use the native fp8/int8 compute paths provided by PyTorch and do not apply this patch.

</details>

---

## First-Run Warmup

**The very first generation after loading a model will be noticeably slower than usual.** This is expected.

On the first forward pass through each eligible layer, the patch converts fp8 weights to INT8 on the fly. After that the converted weights stay in VRAM for the entire session.

- **What is slow:** one-time weight preparation (fp8 → int8, per-channel scale computation). Runs in chunks of 256 rows — peak VRAM overhead is ~2–3 MB per chunk, not ~400 MB.
- **What is not slow:** the GEMM itself. Matrix computation runs at full INT8 speed from the very first call.
- **Subsequent generations** are significantly faster in total wall-clock time. The weight preparation cost is fully amortized after the first run.

---

## What Gets Replaced and What Doesn't

Not every layer is routed through the INT8 kernel. Small and numerically sensitive layers are excluded.

**Excluded from INT8:**
`cap_embedder`, `t_embedder`, `x_embedder`, `cap_pad_token`, `context_refiner`, `final_layer`, `noise_refiner`, `adaLN`, `x_pad_token`

**A layer is eligible for INT8 only when:**
- Weight dtype is `float8_e4m3fn`
- Weight is 2D with `N ≥ 64`, `K ≥ 64`
- `K % 4 == 0` (DP4A packs 4 int8 values into one int32)
- Input tensor is on CUDA
- Input dtype is fp32, fp16, or bf16

All other layers fall through to the stock path unchanged.

---

## 🛠️ Installation (Windows)

For most users, download the pre-compiled `.pyd` file from the [Releases](../../releases) page and skip to step 7.

If you want to build the CUDA extension from source:

**Prerequisites:**
- NVIDIA CUDA Toolkit 12.1+ (tested on 12.8)
- Visual Studio 2022 with the **C++ Desktop Development** workload
- PyTorch 2.6.0+ (tested with cu124)

**Build steps:**

1. Create the extension directory inside your Forge installation:
   ```
   backend/ext/zimage_ext/
   ```

2. Place `zimage_block.cu`, `setup.py`, and any other source files into that folder.

3. Open **x64 Native Tools Command Prompt for VS 2022** (Windows Start Menu → Visual Studio 2022).

4. Navigate to the extension folder:
   ```cmd
   cd path\to\forge\backend\ext\zimage_ext
   ```

5. Set the required environment variable:
   ```cmd
   set DISTUTILS_USE_SDK=1
   ```

6. Build using the Python executable from your Forge virtual environment:
   ```cmd
   ..\..\..\venv\Scripts\python.exe setup.py build_ext --inplace
   ```

7. Replace `operations.py` in the main Forge directory with the patched version.

8. Launch Forge. Done.

---

## Why ~2× Without visible Quality Loss

| Factor | Explanation |
|---|---|
| INT8 arithmetic | `__dp4a` is 2–4× faster than fp16 GEMM on Pascal ALU |
| Narrower memory footprint | int8 = 2× less bandwidth than fp16, 4× less than fp32 |
| Per-channel scale on W | better dequant accuracy than per-tensor |
| Per-token scale on X | dynamic quantization, zero extra latency (fused into epilogue) |
| Excluded layers | numerically sensitive layers (embeddings, norms) stay in fp32 |

Quantization error is `O(scale / 127)` per channel — negligible for weights with well-behaved distributions, which fp8-trained models have by construction.

---

## System Architecture

```
operations.py
└── Linear.forward(x)
        │
        ├─ _can_use_zimage_int8()  → _zimage_int8_linear()   ← this patch (fastest)
        ├─ _can_use_zimage_fp8()   → _zimage_fp8_linear()    ← fp8 GEMM fallback
        └─ torch.nn.functional.linear()                       ← stock fallback
```

The CUDA kernels live in `zimage_block.cu`, compiled as `zimage_ext` — a PyTorch C++ extension loaded at startup from `backend/ext/zimage_ext/`.

---

## CUDA Kernels

<details>
<summary>Kernel 1 — RMSNorm (<code>rms_norm_kernel</code>)</summary>

Fused single-pass RMSNorm: computes row normalization and applies learned weights without a second data pass.

```
rstd = rsqrt( mean(x²) + eps )
out[i] = x[i] * rstd * weight[i]
```

Each thread accumulates `x[i]²` over its stride, then `warp_reduce_sum()` via `__shfl_xor_sync` collapses 32 lanes per warp. Warp leaders write into shared memory; warp 0 does the final cross-warp reduction. `rstd` is broadcast to all threads via `__syncthreads()`.

Block size is rounded up to the nearest power of two (max 1024) at launch time.

</details>

<details>
<summary>Kernel 2 — FP8 GEMM v6 (<code>fp8_gemm_v6_kernel</code>)</summary>

Legacy fallback for layers where INT8 weight preparation hasn't happened yet. Computes `out = X @ W^T` with weights stored in FP8 E4M3, dequantized on the fly into fp32.

**Tile parameters (autotuned for GP102):**

| Constant | Value | Meaning |
|---|---|---|
| BM, BN | 128 | tile rows/cols |
| BK | 8 | reduction tile depth |
| TM, TN | 4, 16 | rows/cols per thread |
| TX × TY | 8 × 32 | block = 256 threads |

**Double-buffered shared memory:**
```
smem_x[2][BM][BK+1]  — activations (fp32)
smem_w[2][BN][BK+1]  — weights (fp32, dequant on-the-fly)
```
`+1` column padding eliminates bank conflicts on 32-bank hardware.

Each tile iteration prefetches the next tile into registers (`prefetch_x[]`, `prefetch_w[]`) while computing on the current smem buffers, then commits before `__syncthreads()`. This hides global memory latency.

**FP8 E4M3 decode** (`fp8_to_f32_v5`) reconstructs the IEEE 754 bit pattern manually:
```c
// exponent bias adjustment: 127_fp32 - 7_e4m3 = 120
result = sign | ((exponent + 120) << 23) | (mantissa << 20)
```

Activation type is templated (`float | __half | bf16`) with overloaded `load_x_to_f32()` inlines.

</details>

<details>
<summary>Kernel 2b — Activation Quantization (<code>quantize_x_int8_kernel</code>)</summary>

Converts fp32/fp16/bf16 activations to INT8 with a **dynamic per-token (per-row) scale** at inference time. One thread block per row, 256 threads.

**Pass 1 — per-row absmax:**
- Each thread finds `max(|x[i]|)` over its elements
- `warp_reduce_max()` collapses to per-warp max
- Warp 0 does final reduce → `scale = absmax / 127.0` (min clamped to `1e-12`)

**Pass 2 — quantize:**
```
q = round(x[i] / scale),  clamped to [-127, 127]
```
Symmetric range `[-127, 127]` (not `-128`) avoids asymmetry issues.

Output: `x_q[M, K]` (int8) + `scale_x[M]` (fp32).

</details>

<details>
<summary>Kernel 2d — INT8 GEMM via DP4A (<code>int8_gemm_dp4a_kernel</code>) — main kernel</summary>

Computes `out[M, N] = X_q[M, K] @ W_q^T[N, K]` with dequantization fused into the epilogue.

**Tile parameters (tuned for 2 blocks/SM on GP102):**

| Constant | Value | Meaning |
|---|---|---|
| I8_BM | 128 | X row tile |
| I8_BN | 64 | W row tile (= output columns) |
| I8_BK | 64 | K tile depth |
| I8_TM | 8 | rows per thread |
| I8_TN | 4 | columns per thread |
| THREADS | 256 | = 16 × 16 |
| KDWORDS | 16 | = BK / 4 (four int8 packed into one int32) |

**Shared memory layout:**
```
smem_x [2][128][17]  int32 — double-buffered  = 17408 bytes
smem_w [1][ 64][17]  int32 — single-buffered  =  4352 bytes
─────────────────────────────────────────────────────────────
Total: 21760 bytes  (<24576 → 2 active blocks/SM on GP102 ✓)
```
The `+1 dword` padding (17 dwords per row) eliminates all bank conflicts.

**Why W uses a single buffer:** W does not need to be re-read after the compute phase. Dropping the second W buffer saves 4352 bytes — exactly enough to fit a second active block per SM on GP102 (26112 → 21760 bytes). W prefetch travels through registers and is committed to `smem_w[0]` after compute finishes.

**The `__dp4a` instruction:**
```c
// Interprets each int32 as 4 packed int8 values.
// Computes dot product of 4 pairs and accumulates into int32.
acc[i][j] = __dp4a(x_reg[i], w_reg[j], acc[i][j]);
```
Per K-tile: 16 `__dp4a` calls × 4 ops each = **64 int8 MACs per thread**, all on regular ALU. Available since sm_61.

**Main loop:**
```
Bootstrap tile 0 → smem_x[0], smem_w[0]
__syncthreads()

for kt in 0..num_tiles:
    buf = kt & 1              // active X buffer (ping-pong)
    nxt = 1 - buf

    if has_next:
        prefetch_x[] ← GMEM x, tile kt+1
        prefetch_w[] ← GMEM w, tile kt+1

    for kk in 0..KDWORDS:
        x_reg[TM] ← smem_x[buf][ty*TM : +TM][kk]
        w_reg[TN] ← smem_w[0][tx + j*TX][kk]
        acc[TM][TN] = __dp4a(x_reg, w_reg, acc)

    if has_next:
        smem_x[nxt] ← prefetch_x[]
        smem_w[0]   ← prefetch_w[]

    __syncthreads()
```

**Epilogue — dequant and write:**
```c
float f_val = (float)acc[i][j] * scale_x[m] * scale_w[n] + bias[n];
out[m * N + n] = cvt_out(f_val, OutT());
// OutT = float  → ZImage
// OutT = half   → Flux
```

**INT32 overflow safety:** worst case at ±127 saturation: `127 × 127 × K = 16129 × K`. For K ≤ 130,000: max ≈ `2.1 × 10⁹ < INT32_MAX` — safe.

</details>

<details>
<summary>Kernel 3 — RoPE (<code>apply_rope_kernel</code>)</summary>

Applies Rotary Position Embeddings in-place to Q and K. GQA-compatible: Q is rotated for all `nh` heads, K only for `nkh`.

Each `blockIdx.x` maps to one token-head vector; each `threadIdx.x` handles one `(x₀, x₁)` pair:
```
[q₀']   [cos  -sin] [q₀]
[q₁'] = [sin   cos] [q₁]
```

</details>

---

## Python Integration (`operations.py`)

<details>
<summary>Per-layer INT8 state and lazy weight preparation</summary>

**Per-layer attributes:**

| Attribute | Type | Purpose |
|---|---|---|
| `weight_int8` | `int8 [N, K]` | converted INT8 weights |
| `weight_scale_zi` | `float32 [N]` | per-channel dequant scale |
| `_zi_int8_ready` | bool | True after first conversion |
| `_zi_int8_eligible` | bool | passed size/dtype gate checks |
| `_zi_int8_eligibility_checked` | bool | lazy check already ran |

**Lazy weight preparation** (called once per layer on the first eligible forward pass):

```python
for chunk in range(0, N, 256):          # ~2–3 MB peak VRAM per chunk
    fp32 = w_fp8[chunk].to(float32)
    absmax = fp32.abs().amax(dim=1)
    scale = (absmax / 127.0).clamp(min=1e-12)
    w_int8[chunk] = (fp32 / scale).round().clamp(-127, 127).to(int8)
    scale_zi[chunk] = scale

layer.weight = None      # free the original fp8 tensor from VRAM
layer._zi_int8_ready = True
```

Chunked processing avoids materializing a full fp32 copy of the weight matrix. For a 4096×4096 layer that would be ~400 MB; with chunking it stays at a few MB at any point.

</details>

<details>
<summary>Debug counters</summary>

`_zi_dbg` tracks per-session statistics for every routing decision:
```python
{
    "fwd_i8_ok":       ...,  # successful INT8 DP4A forwards
    "fwd_fp8_ok":      ...,  # fp8 GEMM fallback forwards
    "fwd_fallback":    ...,  # stock torch fallback forwards
    "i8_prequant":     ...,  # layers converted to INT8 this session
    "fwd_i8_err":      ...,  # INT8 errors (logged with full traceback)
    ...
}
```

`_zi_dump_stats()` is auto-hooked onto `backend.sampling.sample` and prints a summary after each generation.

</details>
