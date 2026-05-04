"""
zimage_ext/__init__.py — Python API for the CUDA extension

Provides drop-in replacements for:
    rms_norm(x, weight, eps)          → replaces torch.nn.functional.rms_norm
    fp8_gemm(x, w, scale)             → replaces Triton fp8 GEMM
    flash_attn(q, k, v, scale)        → replaces F.scaled_dot_product_attention

Auto-loads the compiled .pyd. Falls back gracefully if not compiled yet.
"""

import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)

# ── Try loading the compiled extension ───────────────────────────────────────

_ext = None
_AVAILABLE = False

def _load():
    global _ext, _AVAILABLE
    try:
        # First try: already in sys.path (installed via pip install -e .)
        import zimage_ext as _m
        _ext = _m
        _AVAILABLE = True
        logger.info("zimage_ext CUDA extension loaded.")
        return
    except ImportError:
        pass

    # Second try: load from the directory of this file (build_ext --inplace)
    _dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _dir)
    try:
        import zimage_ext as _m
        _ext = _m
        _AVAILABLE = True
        logger.info("zimage_ext CUDA extension loaded (local).")
        return
    except ImportError:
        pass
    finally:
        sys.path.pop(0)

    logger.warning(
        "zimage_ext not compiled. Run: cd backend/ext/zimage_ext && pip install -e ."
    )

_load()


# ── Public API ────────────────────────────────────────────────────────────────

def is_available() -> bool:
    return _AVAILABLE


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused RMSNorm. Single CUDA kernel, no intermediate allocations.
    Replaces: torch.nn.functional.rms_norm(x, (N,), weight, eps)
    """
    if _AVAILABLE and x.is_cuda and x.dtype == torch.float32:
        return _ext.rms_norm(x, weight, eps)
    # Fallback
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)


def fp8_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    scale_w: float,
) -> torch.Tensor:
    """
    fp8 dequant + GEMM.
    x: [M, K] float32
    w: [N, K] float8_e4m3fn (or uint8)
    Returns: [M, N] float32
    """
    if _AVAILABLE and x.is_cuda:
        return _ext.fp8_gemm(x, w, scale_w)
    # Fallback: dequant on CPU then matmul (slow, for debugging only)
    w_f32 = w.float() * scale_w
    return torch.nn.functional.linear(x, w_f32)


def flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float = None,
) -> torch.Tensor:
    """
    Flash Attention for sm61. No [seq, seq] matrix in VRAM.
    q, k, v: [B, nh, seq, hd] float32
    Returns:  [B, nh, seq, hd] float32
    """
    import math
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    if _AVAILABLE and q.is_cuda and q.dtype == torch.float32:
        return _ext.flash_attn(q, k, v, softmax_scale)
    # Fallback
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=softmax_scale
    )