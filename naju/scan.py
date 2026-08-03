"""Scan backend dispatch + pure-PyTorch reference for the Naju v1 recurrence.

All backends share one signature and are numerically interchangeable:

    y = scan(u, f_logit, i_logit, B, C, D_skip)

    u        [B, T, D]   inner stream (post conv+SiLU)
    f_logit  [B, T, D]   forget-gate pre-activation (sigmoid applied inside)
    i_logit  [B, T, D]   input-gate pre-activation
    B, C     [B, T, N]   input-dependent write/read vectors (N = d_state)
    D_skip   [D]         feedthrough weight (pass zeros to get the pure state
                         readout cx = sum_n C_n x_n; the mixer applies its own
                         D skip outside the scan)

Recurrence (per batch b, channel d, state n; sigmas are elementwise):

    f_t = sigmoid(f_logit_t)          i_t = sigmoid(i_logit_t)
    x_t[n] = f_t * x_{t-1}[n] + i_t * B_t[n] * u_t
    y_t    = sum_n C_t[n] * x_t[n] + D_skip * u_t

Backends — one per role, no overlap:
    reference : this file, sequential pure PyTorch. Exact, any device. Slow.
                Ground truth for all numerical checks.
    chunk     : chunk_triton.py — fused chunk-parallel (SSD-style) Triton
                kernels; exact for f_logit >= -5, auto-falls back to `cuda`
                below that. TRAINING standard.
    cuda      : cuda/ — JIT-compiled native CUDA kernel, register-resident
                state + recompute backward. Exact for any logit; the `chunk`
                guard fallback and the memory-light long-T training path.
    cuda_bw   : cuda/bw.py — warp-shuffle variant. INFERENCE standard
                (the paper's efficiency tables use this backend under no_grad).
    affine    : chunk_affine_triton.py — hierarchical affine chunk scan
                (fused_affine_chunk). Forward exact for any f_logit (direct
                fp32 products, no decay ratios); training keeps the `chunk`
                guard because it shares the same fused backward. EXPERIMENTAL.

Selection order: explicit `backend` argument > NAJU_SCAN_BACKEND env >
auto ("chunk" when CUDA is available, else "reference").
"""
import os

import torch

_BACKEND_CACHE = {}


def resolve_backend(backend=None):
    b = backend or os.environ.get("NAJU_SCAN_BACKEND")
    if not b:
        b = "chunk" if torch.cuda.is_available() else "reference"
    return b.lower()


def _get(b):
    fn = _BACKEND_CACHE.get(b)
    if fn is None:
        if b == "reference":
            fn = naju_scan_reference
        elif b == "chunk":
            from naju.chunk_triton import naju_chunk_scan as fn
        elif b == "cuda":
            from naju.cuda import naju_cuda_scan as fn
        elif b == "cuda_bw":
            from naju.cuda.bw import naju_cuda_bw_scan as fn
        elif b in ("affine", "fused_affine_chunk"):
            from naju.chunk_affine_triton import naju_affine_chunk_scan as fn
        else:
            raise ValueError(f"unknown scan backend {b!r}; "
                             f"valid: reference, chunk, cuda, cuda_bw, affine")
        _BACKEND_CACHE[b] = fn
    return fn


def naju_scan(u, f_logit, i_logit, B, C, D_skip, backend=None):
    return _get(resolve_backend(backend))(u, f_logit, i_logit, B, C, D_skip)


def naju_scan_reference(u, f_logit, i_logit, B, C, D_skip):
    """Sequential reference implementation (exact, any device)."""
    B_, T, D = u.shape
    f = torch.sigmoid(f_logit)
    i = torch.sigmoid(i_logit)
    x = u.new_zeros(B_, D, B.shape[-1])
    ys = []
    for t in range(T):
        f_t = f[:, t].unsqueeze(-1); i_t = i[:, t].unsqueeze(-1)
        B_t = B[:, t].unsqueeze(1);  C_t = C[:, t].unsqueeze(1)
        u_t = u[:, t].unsqueeze(-1)
        x = f_t * x + i_t * B_t * u_t
        y_t = (C_t * x).sum(-1) + D_skip.unsqueeze(0) * u[:, t]
        ys.append(y_t)
    return torch.stack(ys, dim=1)
