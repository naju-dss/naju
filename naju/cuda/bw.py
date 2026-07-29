"""Warp-shuffle-optimized CUDA scan wrapper (naju_cuda_bw.cu).

Same signature as naju.cuda.naju_cuda_scan; select via scan_backend
="cuda_bw" (or NAJU_SCAN_BACKEND=cuda_bw). Fastest inference path (2.5-3.2x
over the base CUDA scan); the paper's efficiency tables use this backend,
measured forward-only under no_grad.

TORCH_CUDA_ARCH_LIST is intentionally not set: torch auto-detects the local
GPU arch.
"""
import os
import torch
from torch.utils.cpp_extension import load

_THIS = os.path.dirname(os.path.abspath(__file__))
_ext = None


def _mod():
    global _ext
    if _ext is None:
        _ext = load(
            name="naju_cuda_bw",
            sources=[os.path.join(_THIS, "naju_cuda_bw.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=os.environ.get("NAJU_CUDA_VERBOSE", "0") == "1",
        )
    return _ext


class _BWFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, f, i, B, C, D):
        ext = _mod()
        u, f, i, B, C, D = [t.contiguous().float() for t in (u, f, i, B, C, D)]
        save = any(ctx.needs_input_grad)
        if not save:
            return ext.fwd_chunked(u, f, i, B, C, D)
        y, ckpt = ext.fwd(u, f, i, B, C, D, save)
        ctx.save_for_backward(u, f, i, B, C, D, ckpt)
        return y

    @staticmethod
    def backward(ctx, dy):
        ext = _mod()
        u, f, i, B, C, D, ckpt = ctx.saved_tensors
        return tuple(ext.bwd(u, f, i, B, C, D, ckpt, dy.contiguous().float()))


def naju_cuda_bw_scan(u, f_logit, i_logit, B, C, D_skip):
    return _BWFn.apply(u, f_logit, i_logit, B, C, D_skip)
