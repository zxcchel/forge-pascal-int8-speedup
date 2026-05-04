/*
 * zimage_block.cpp — PyTorch C++ bindings v8
 *
 *   Старые ops:
 *     zimage_ext.rms_norm(x, weight, eps)
 *     zimage_ext.fp8_gemm(x, w, scale_w)            ← FP32 fallback path
 *     zimage_ext.flash_attn(q, k, v, softmax_scale)
 *
 *   Новые INT8 ops:
 *     zimage_ext.quantize_weight_int8(w_fp8)        → (w_int8 [N,K], scale_w [N])
 *       Вызывается ОДИН РАЗ при загрузке модели, на CPU или GPU.
 *     zimage_ext.int8_linear(x, w_int8, scale_w, bias=None)
 *       Вызывается на каждом forward. Внутренне делает:
 *         x_q, scale_x = quantize_x_int8(x)        ← per-token, на лету
 *         out = int8_gemm(x_q, w_int8, scale_x, scale_w, bias)
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

extern "C" {
    void launch_rms_norm(
        const float*, const float*, float*,
        int, int, float, cudaStream_t);

    void launch_fp8_gemm(
        const float*, const uint8_t*, float*,
        float, int, int, int, cudaStream_t);
    void launch_fp8_gemm_f16(
        const void*, const uint8_t*, float*,
        float, int, int, int, cudaStream_t);
    void launch_fp8_gemm_bf16(
        const void*, const uint8_t*, float*,
        float, int, int, int, cudaStream_t);

    void launch_quantize_x_f32(const void*, int8_t*, float*, int, int, cudaStream_t);
    void launch_quantize_x_f16(const void*, int8_t*, float*, int, int, cudaStream_t);
    void launch_quantize_x_bf16(const void*, int8_t*, float*, int, int, cudaStream_t);

    void launch_int8_gemm_f32(
        const int8_t*, const int8_t*,
        const float*, const float*,
        const float*, float*,
        int, int, int, cudaStream_t);

    void launch_int8_gemm_f16(
        const int8_t*, const int8_t*,
        const float*, const float*,
        const float*, void*,
        int, int, int, cudaStream_t);

    void launch_flash_attn(
        const float*, const float*, const float*, float*,
        float, int, int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        cudaStream_t);
}

// ---------------------------------------------------------------------------
// rms_norm
// ---------------------------------------------------------------------------
torch::Tensor rms_norm_fwd(
    torch::Tensor x,
    torch::Tensor weight,
    double eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    int N = x.size(-1);
    int rows = x.numel() / N;

    x = x.contiguous();
    weight = weight.contiguous();

    auto out = torch::empty_like(x);
    auto stream = c10::cuda::getCurrentCUDAStream();

    launch_rms_norm(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        out.data_ptr<float>(),
        rows, N, (float)eps,
        stream);

    return out;
}

// ---------------------------------------------------------------------------
// fp8_gemm — FP32 fallback path (без изменений)
// ---------------------------------------------------------------------------
torch::Tensor fp8_gemm_fwd(
    torch::Tensor x,
    torch::Tensor w,
    double scale_w)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    TORCH_CHECK(w.is_cuda(), "w must be CUDA tensor");
    TORCH_CHECK(w.dtype() == torch::kUInt8 || w.dtype() == torch::kFloat8_e4m3fn,
        "w must be uint8 or float8_e4m3fn");

    auto x_dtype = x.scalar_type();
    TORCH_CHECK(
        x_dtype == torch::kFloat32 ||
        x_dtype == torch::kFloat16 ||
        x_dtype == torch::kBFloat16,
        "x must be float32, float16, or bfloat16");
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dimensions");

    int K = x.size(-1);
    int N = w.size(0);
    TORCH_CHECK(w.size(1) == K, "w.size(1) must equal x.size(-1)");

    auto out_shape = x.sizes().vec();
    out_shape.back() = N;

    auto x2d = x.reshape({ -1, K }).contiguous();
    int M = x2d.size(0);

    w = w.contiguous();

    auto out2d = torch::empty({ M, N },
        torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto wptr = reinterpret_cast<const uint8_t*>(w.data_ptr());

    if (x_dtype == torch::kFloat32) {
        launch_fp8_gemm(x2d.data_ptr<float>(), wptr,
            out2d.data_ptr<float>(), (float)scale_w, M, N, K, stream);
    }
    else if (x_dtype == torch::kFloat16) {
        launch_fp8_gemm_f16(x2d.data_ptr(), wptr,
            out2d.data_ptr<float>(), (float)scale_w, M, N, K, stream);
    }
    else {
        launch_fp8_gemm_bf16(x2d.data_ptr(), wptr,
            out2d.data_ptr<float>(), (float)scale_w, M, N, K, stream);
    }

    return out2d.reshape(out_shape);
}

// ---------------------------------------------------------------------------
// quantize_weight_int8
//
//   Вход:  w   [N, K]  fp8_e4m3 (или uint8 raw bytes), CUDA или CPU
//   Выход: (w_int8 [N, K] int8, scale_w [N] fp32) — оба на CUDA
//
//   Делается ОДИН РАЗ при загрузке модели. Per-channel (по N) absmax квантизация.
//   Реализация на стороне Python через torch ops — потому что:
//     1) этот путь не критичен по скорости (один раз на запуск)
//     2) не нужен отдельный kernel ради 5K вызовов на старте
//
//   Логика тут только: dequant fp8→fp32 + absmax по K + scale + round.
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> quantize_weight_int8(torch::Tensor w)
{
    TORCH_CHECK(w.dim() == 2, "w must be 2D [N, K]");
    TORCH_CHECK(w.dtype() == torch::kUInt8 || w.dtype() == torch::kFloat8_e4m3fn ||
        w.dtype() == torch::kFloat32 || w.dtype() == torch::kFloat16 ||
        w.dtype() == torch::kBFloat16,
        "w must be fp8, uint8, fp32, fp16, or bf16");

    auto device = w.device();
    if (!device.is_cuda()) {
        // Перенесём на CUDA для скорости quantize, обратно класть будем уже
        // как готовый int8 + scale.
        device = torch::Device(torch::kCUDA, 0);
    }

    // Dequantize fp8 → fp32 (если нужно) через наш FP32 GEMM path? Нет, проще
    // просто привести через PyTorch (на старте это не критично).
    torch::Tensor w_fp32;
    if (w.dtype() == torch::kFloat8_e4m3fn) {
        w_fp32 = w.to(device, torch::kFloat32);
    }
    else if (w.dtype() == torch::kUInt8) {
        // uint8 это raw fp8 bytes — нужно правильно интерпретировать
        auto w_fp8 = w.to(device).view(torch::kFloat8_e4m3fn);
        w_fp32 = w_fp8.to(torch::kFloat32);
    }
    else {
        w_fp32 = w.to(device, torch::kFloat32);
    }

    // Per-channel absmax по K (dim=1) → scale [N]
    auto absmax = w_fp32.abs().amax(/*dim=*/1);  // [N]
    auto scale = absmax / 127.0f;
    // Защита от нулевых rows
    scale = torch::clamp_min(scale, 1e-12f);

    // Quantize: round(w / scale[:, None])
    auto w_scaled = w_fp32 / scale.unsqueeze(1);
    auto w_int8 = w_scaled.round().clamp(-127.0f, 127.0f).to(torch::kInt8);

    return std::make_tuple(w_int8.contiguous(), scale.to(torch::kFloat32).contiguous());
}

// ---------------------------------------------------------------------------
// int8_linear
//
//   x        : [..., K]  fp32 / fp16 / bf16, CUDA
//   w_int8   : [N, K]    int8, CUDA (precomputed)
//   scale_w  : [N]       fp32, CUDA (precomputed)
//   bias     : [N] or None, любой dtype
//
//   Возврат: [..., N] в dtype'е x
//
//   Внутри:
//     1) Reshape x → 2D [M, K]
//     2) quantize_x: x → x_int8 [M, K] + scale_x [M]
//     3) int8_gemm: out_fp32 = (x_int8 @ w_int8.T) * scale_x[m] * scale_w[n] + bias[n]
//     4) Cast в dtype(x), reshape назад
// ---------------------------------------------------------------------------
torch::Tensor int8_linear_fwd(
    torch::Tensor x,
    torch::Tensor w_int8,
    torch::Tensor scale_w,
    c10::optional<torch::Tensor> bias_opt)
{
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w_int8.is_cuda(), "w_int8 must be CUDA");
    TORCH_CHECK(scale_w.is_cuda(), "scale_w must be CUDA");
    TORCH_CHECK(w_int8.dtype() == torch::kInt8, "w_int8 must be int8");
    TORCH_CHECK(scale_w.dtype() == torch::kFloat32, "scale_w must be float32");

    auto x_dtype = x.scalar_type();
    TORCH_CHECK(
        x_dtype == torch::kFloat32 ||
        x_dtype == torch::kFloat16 ||
        x_dtype == torch::kBFloat16,
        "x must be float32, float16, or bfloat16");

    TORCH_CHECK(x.dim() >= 2, "x must be at least 2D");
    TORCH_CHECK(w_int8.dim() == 2, "w_int8 must be 2D");

    int K = x.size(-1);
    int N = w_int8.size(0);
    TORCH_CHECK(w_int8.size(1) == K, "w_int8.size(1) must equal x.size(-1)");
    TORCH_CHECK(scale_w.numel() == N, "scale_w must have N elements");

    TORCH_CHECK((K % 4) == 0, "K must be divisible by 4 for DP4A path");

    auto out_shape = x.sizes().vec();
    out_shape.back() = N;

    auto x2d = x.reshape({ -1, K }).contiguous();
    int M = x2d.size(0);

    w_int8 = w_int8.contiguous();
    scale_w = scale_w.contiguous();

    auto opts_i8 = torch::TensorOptions().dtype(torch::kInt8).device(x.device());
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());

    auto x_int8 = torch::empty({ M, K }, opts_i8);
    auto scale_x = torch::empty({ M }, opts_f32);
    // ВАЖНО: Мы больше не выделяем out2d здесь. Мы сделаем это ниже в ветках.

    auto stream = c10::cuda::getCurrentCUDAStream();

    // ── Step 1: quantize x ─────────────────────────────────────────────
    if (x_dtype == torch::kFloat32) {
        launch_quantize_x_f32(x2d.data_ptr(),
            x_int8.data_ptr<int8_t>(), scale_x.data_ptr<float>(),
            M, K, stream);
    }
    else if (x_dtype == torch::kFloat16) {
        launch_quantize_x_f16(x2d.data_ptr(),
            x_int8.data_ptr<int8_t>(), scale_x.data_ptr<float>(),
            M, K, stream);
    }
    else {
        launch_quantize_x_bf16(x2d.data_ptr(),
            x_int8.data_ptr<int8_t>(), scale_x.data_ptr<float>(),
            M, K, stream);
    }

    // ── Step 2: preparation for GEMM ───────────────────────────────────
    const float* bias_ptr = nullptr;
    torch::Tensor bias_f32;
    if (bias_opt.has_value()) {
        auto& b = bias_opt.value();
        TORCH_CHECK(b.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(b.numel() == N, "bias size mismatch");
        bias_f32 = b.to(torch::kFloat32).contiguous();
        bias_ptr = bias_f32.data_ptr<float>();
    }

    // ── Step 3: Branching by type (Allocate and Compute) ───────────────
    if (x_dtype == torch::kFloat32) {
        // Путь Z-IMAGE: Выделяем FP32 и считаем (быстро, как раньше)
        auto out2d = torch::empty({ M, N }, opts_f32);

        launch_int8_gemm_f32(
            x_int8.data_ptr<int8_t>(),
            w_int8.data_ptr<int8_t>(),
            scale_x.data_ptr<float>(),
            scale_w.data_ptr<float>(),
            bias_ptr,
            out2d.data_ptr<float>(),
            M, N, K, stream);

        return out2d.reshape(out_shape);

    }
    else if (x_dtype == torch::kFloat16) {
        // Путь FLUX: Выделяем FP16, экономя половину памяти!
        auto opts_f16 = torch::TensorOptions().dtype(torch::kFloat16).device(x.device());
        auto out2d = torch::empty({ M, N }, opts_f16);

        launch_int8_gemm_f16(
            x_int8.data_ptr<int8_t>(),
            w_int8.data_ptr<int8_t>(),
            scale_x.data_ptr<float>(),
            scale_w.data_ptr<float>(),
            bias_ptr,
            out2d.data_ptr(), // Передаем void* 
            M, N, K, stream);

        return out2d.reshape(out_shape);

    }
    else {
        // Путь BFloat16 (Fallback): Выделяем FP32, потом кастим в PyTorch
        auto out2d = torch::empty({ M, N }, opts_f32);

        launch_int8_gemm_f32(
            x_int8.data_ptr<int8_t>(),
            w_int8.data_ptr<int8_t>(),
            scale_x.data_ptr<float>(),
            scale_w.data_ptr<float>(),
            bias_ptr,
            out2d.data_ptr<float>(),
            M, N, K, stream);

        return out2d.to(x_dtype).reshape(out_shape);
    }
}

// ---------------------------------------------------------------------------
// flash_attn
// ---------------------------------------------------------------------------
torch::Tensor flash_attn_fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double softmax_scale)
{
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(q.dtype() == torch::kFloat32, "inputs must be float32");
    TORCH_CHECK(q.dim() == 4, "q must be [B, nh, seq, hd]");

    // Форсируем contiguous ДО передачи в ядро.
    //
    // После permute(0,2,1,3) q/k/v имеют stride_kn = nh*hd (≈3072 для 24 головы).
    // Строки K тогда расположены через 3072*4 = 12 KB — каждая в отдельной строке
    // DRAM row buffer (8 KB), что убивает эффективную пропускную способность с 484
    // до ~100 GB/s. С contiguous stride_kn = hd = 128 → строки через 512 байт →
    // 16 строк в одном DRAM row → пиковая пропускная способность.
    //
    // Стоимость копии: 3 × [B,nh,M,hd] × fp32 ≈ 150 MB на вызов (≈0.3ms при 484 GB/s)
    // vs. выигрыш от правильного доступа: ~5× меньше задержек DRAM.
    q = q.contiguous();
    k = k.contiguous();
    v = v.contiguous();

    int B = q.size(0);
    int nh = q.size(1);
    int M = q.size(2);
    int hd = q.size(3);
    int N = k.size(2);

    TORCH_CHECK(hd == 64 || hd == 128,
        "head_dim must be 64 or 128, got ", hd);

    // Output всегда contiguous [B, nh, M, hd] — q после .contiguous() уже такой.
    auto out = torch::empty({ B, nh, M, hd },
        torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));
    auto stream = c10::cuda::getCurrentCUDAStream();

    // Сигнатура launch_flash_attn не менялась — stride(3) передаём как раньше,
    // внутри launcher'а он помечен [[maybe_unused]] / закомментирован.
    launch_flash_attn(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        (float)softmax_scale,
        B, nh, M, N, hd,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        stream);

    return out;
}


// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ZImage CUDA extension v8 — INT8 DP4A GEMM for Lumina NextDiT on sm61";

    m.def("rms_norm", &rms_norm_fwd,
        "Fused RMSNorm",
        py::arg("x"), py::arg("weight"), py::arg("eps") = 1e-6);

    m.def("fp8_gemm", &fp8_gemm_fwd,
        "FP8 dequant + GEMM (FP32 fallback)",
        py::arg("x"), py::arg("w"), py::arg("scale_w"));

    m.def("quantize_weight_int8", &quantize_weight_int8,
        "Pre-quantize a weight tensor for INT8 path. Call ONCE at model load.\n"
        "Args:\n"
        "  w: [N, K] fp8/fp16/bf16/fp32 weight\n"
        "Returns:\n"
        "  (w_int8 [N, K] int8, scale_w [N] fp32)",
        py::arg("w"));

    m.def("int8_linear", &int8_linear_fwd,
        "INT8 GEMM via DP4A. Quantizes activations on-the-fly (per-token).\n"
        "Args:\n"
        "  x:        [..., K] fp32/fp16/bf16\n"
        "  w_int8:   [N, K] int8 (precomputed)\n"
        "  scale_w:  [N] fp32 (precomputed)\n"
        "  bias:     [N] or None\n"
        "Returns:\n"
        "  [..., N] in x.dtype",
        py::arg("x"), py::arg("w_int8"), py::arg("scale_w"),
        py::arg("bias") = py::none());

    m.def("flash_attn", &flash_attn_fwd,
        "Flash Attention fwd",
        py::arg("q"), py::arg("k"), py::arg("v"),
        py::arg("softmax_scale") = -1.0);
}