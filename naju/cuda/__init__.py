"""Native CUDA scan for the Naju v1 recurrence (forward + recompute backward).

Drop-in replacement for the reference scan with an identical signature:

    from naju.cuda import naju_cuda_scan
    y = naju_cuda_scan(u, f_logit, i_logit, B, C, D_skip)

The extension is JIT-compiled on first use via torch.utils.cpp_extension.load
(no separate build step). Forward keeps the recurrent state register-resident and
never materializes the [B,T,D,N] history; for training it checkpoints the state
every CK steps and the backward recomputes each chunk (Mamba-style), so peak
memory stays close to the inference footprint.

TORCH_CUDA_ARCH_LIST is intentionally not set here: torch auto-detects the
local GPU arch. Set the env var explicitly only for cross-compilation.
"""
import os

import torch
from torch.utils.cpp_extension import load

_THIS = os.path.dirname(os.path.abspath(__file__))
_ext = None


def _ext_mod():
    global _ext
    if _ext is None:
        _ext = load(
            name="naju_cuda",
            sources=[os.path.join(_THIS, "naju_cuda.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=os.environ.get("NAJU_CUDA_VERBOSE", "0") == "1",
        )
    return _ext


class _NajuCudaFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, f_logit, i_logit, B, C, D_skip):
        ext = _ext_mod()
        u = u.contiguous().float()
        f_logit = f_logit.contiguous().float()
        i_logit = i_logit.contiguous().float()
        B = B.contiguous().float()
        C = C.contiguous().float()
        D_skip = D_skip.contiguous().float()
        save_ckpt = any(ctx.needs_input_grad)   # False under no_grad (inference)
        if not save_ckpt:
            # inference: chunk-parallel forward (time-axis parallelism)
            return ext.fwd_chunked(u, f_logit, i_logit, B, C, D_skip)
        y, ckpt = ext.fwd(u, f_logit, i_logit, B, C, D_skip, save_ckpt)
        ctx.save_for_backward(u, f_logit, i_logit, B, C, D_skip, ckpt)
        return y

    @staticmethod
    def backward(ctx, dy):
        ext = _ext_mod()
        u, f_logit, i_logit, B, C, D_skip, ckpt = ctx.saved_tensors
        du, df, di, dB, dC, dD = ext.bwd(
            u, f_logit, i_logit, B, C, D_skip, ckpt, dy.contiguous().float()
        )
        return du, df, di, dB, dC, dD


def naju_cuda_scan(u, f_logit, i_logit, B, C, D_skip):
    return _NajuCudaFn.apply(u, f_logit, i_logit, B, C, D_skip)
