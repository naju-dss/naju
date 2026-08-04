"""Hierarchical affine chunk scan (fused_affine_chunk) for the Naju v1 recurrence.

Per-token affine pair a_t = (f_t, w_t), w_t = i_t*B_t*u_t, composed as
(f2,w2)∘(f1,w1) = (f2*f1, w2 + f2*w1). Three phases, no Q×Q retention matrix
and no decay-ratio exp/log math (direct fp32 products only, so the FORWARD is
exact for any f_logit — unlike chunk_triton's f_logit >= -5 envelope):

  Phase A (Triton): per (batch, chunk, channel-tile) register-resident local
          scan from zero state. Emits the partial readout cz_t = C_t·z_t and
          prefix retention P_t (both [B,T,D], same traffic as chunk_triton's
          intra/eL), plus chunk summaries F_c [B,K,D], W_c [B,K,D,N].
  Phase B (Triton): exclusive affine scan over the K chunk summaries →
          chunk-start states. Reuses chunk_bwd_triton._affine_scan_kernel
          (register-resident, one launch — replaces chunk_triton's K-step
          Python carry loop, the launch bottleneck at long T).
  Phase C (Triton): y = cz + P * (C @ s_c^T) + D_skip*u fused in one kernel
          (tl.dot for the boundary correction — replaces chunk_triton's
          einsum + separate elementwise add, one fewer [B,T,D] round trip).

Backward: reuses the backend-agnostic v4 fused backward
(chunk_bwd_triton.naju_chunk_backward) with THIS forward injected for its
time-reversed dw pass. Its SB=16 decay math keeps the same f_logit >= -5
training guard as the chunk backend; inference (no grad) skips the guard
entirely since this forward needs no envelope.
"""
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _affine_local_kernel(u_ptr, fl_ptr, il_ptr, B_ptr, C_ptr,
                         cz_ptr, P_ptr, F_ptr, W_ptr,
                         T, D, N,
                         Q: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr):
    """Phase A: local scan z_q = f_q*z_{q-1} + i_q*u_q*B_q from zero, state
    [BD, BN] in registers; per-token stores are the [BD] reductions only."""
    pid_bk = tl.program_id(0)          # fused (batch, chunk)
    pid_d = tl.program_id(1)           # channel tile
    K = T // Q
    b = pid_bk // K
    k = pid_bk % K

    offs_d = pid_d * BD + tl.arange(0, BD)
    offs_n = tl.arange(0, BN)
    dm = offs_d < D
    nm = offs_n < N

    z = tl.zeros((BD, BN), dtype=tl.float32)
    p = tl.zeros((BD,), dtype=tl.float32) + 1.0
    for q in range(Q):
        t = k * Q + q
        td = (b * T + t) * D + offs_d
        fl = tl.load(fl_ptr + td, mask=dm, other=0.0).to(tl.float32)
        il = tl.load(il_ptr + td, mask=dm, other=0.0).to(tl.float32)
        uu = tl.load(u_ptr + td, mask=dm, other=0.0).to(tl.float32)
        bn = (b * T + t) * N + offs_n
        Bt = tl.load(B_ptr + bn, mask=nm, other=0.0).to(tl.float32)
        Ct = tl.load(C_ptr + bn, mask=nm, other=0.0).to(tl.float32)

        f = tl.sigmoid(fl)
        w = tl.sigmoid(il) * uu                       # [BD]
        z = f[:, None] * z + w[:, None] * Bt[None, :]
        p = p * f
        cz = tl.sum(Ct[None, :] * z, axis=1)          # [BD]
        tl.store(cz_ptr + td, cz.to(cz_ptr.dtype.element_ty), mask=dm)
        tl.store(P_ptr + td, p.to(P_ptr.dtype.element_ty), mask=dm)

    tl.store(F_ptr + (b * K + k) * D + offs_d, p, mask=dm)
    w_off = ((b * K + k) * D + offs_d)[:, None] * N + offs_n[None, :]
    tl.store(W_ptr + w_off, z, mask=dm[:, None] & nm[None, :])


@triton.jit
def _affine_correct_kernel(u_ptr, cz_ptr, P_ptr, C_ptr, S0_ptr, Dskip_ptr,
                           y_ptr, T, D, N,
                           Q: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr):
    """Phase C: y = cz + P * (C @ s_c^T) + D_skip*u, one fused pass."""
    pid_bk = tl.program_id(0)
    pid_d = tl.program_id(1)
    K = T // Q
    b = pid_bk // K
    k = pid_bk % K

    offs_q = tl.arange(0, Q)
    offs_d = pid_d * BD + tl.arange(0, BD)
    offs_n = tl.arange(0, BN)
    dm = offs_d < D
    nm = offs_n < N

    s_off = ((b * K + k) * D + offs_d)[:, None] * N + offs_n[None, :]
    S0 = tl.load(S0_ptr + s_off, mask=dm[:, None] & nm[None, :], other=0.0)
    bc_off = (b * T + k * Q + offs_q)[:, None] * N + offs_n[None, :]
    Cc = tl.load(C_ptr + bc_off, mask=nm[None, :], other=0.0).to(tl.float32)

    corr = tl.dot(Cc, tl.trans(S0), allow_tf32=False)          # [Q, BD]

    td = (b * T + k * Q + offs_q)[:, None] * D + offs_d[None, :]
    m2 = dm[None, :]
    cz = tl.load(cz_ptr + td, mask=m2, other=0.0).to(tl.float32)
    P = tl.load(P_ptr + td, mask=m2, other=0.0).to(tl.float32)
    uu = tl.load(u_ptr + td, mask=m2, other=0.0).to(tl.float32)
    Dk = tl.load(Dskip_ptr + offs_d, mask=dm, other=0.0).to(tl.float32)

    y = cz + P * corr + Dk[None, :] * uu
    tl.store(y_ptr + td, y.to(y_ptr.dtype.element_ty), mask=m2)


class _AffineChunkScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, f_logit, i_logit, B, C, D_skip, Q):
        B_, T, D = u.shape
        N = B.shape[-1]
        K = T // Q
        # bf16/fp16 inputs stay in their dtype (loads upconvert in-kernel,
        # recurrence/summaries accumulate fp32) — halves the dominant Phase A
        # traffic under autocast training. fp32 inputs behave as before.
        dt = u.dtype if u.dtype in (torch.bfloat16, torch.float16) else torch.float32
        uc, flc, ilc = (t.contiguous().to(dt) for t in (u, f_logit, i_logit))
        Bc, Cc = B.contiguous().to(dt), C.contiguous().to(dt)

        cz = torch.empty(B_, T, D, device=u.device, dtype=dt)
        P = torch.empty(B_, T, D, device=u.device, dtype=dt)
        Fc = torch.empty(B_, K, D, device=u.device, dtype=torch.float32)
        Wc = torch.empty(B_, K, D, N, device=u.device, dtype=torch.float32)
        BD = 64
        BN = max(16, triton.next_power_of_2(N))
        grid = (B_ * K, triton.cdiv(D, BD))
        # num_warps=2: the [BD,BN] register tile per program wants few, fat
        # warps (64 elems/thread) — 4/8 warps spill and run ~3x slower
        _affine_local_kernel[grid](uc, flc, ilc, Bc, Cc, cz, P, Fc, Wc,
                                   T, D, N, Q=Q, BD=BD, BN=BN, num_warps=2)

        from naju.chunk_bwd_triton import _exclusive_scan
        x_start = _exclusive_scan(Fc, Wc)             # state entering chunk k

        y = torch.empty(B_, T, D, device=u.device, dtype=dt)
        _affine_correct_kernel[grid](uc, cz, P, Cc, x_start,
                                     D_skip.contiguous().to(dt), y,
                                     T, D, N, Q=Q, BD=BD, BN=BN)
        ctx.save_for_backward(u, f_logit, i_logit, B, C, D_skip)
        ctx.Q = Q
        return y.to(u.dtype)

    @staticmethod
    def backward(ctx, dy):
        from naju.chunk_bwd_triton import naju_chunk_backward
        u, f_logit, i_logit, B, C, D_skip = ctx.saved_tensors

        def _fwd(u2, fl2, il2, B2, C2, D2):
            return _AffineChunkScan.apply(u2, fl2, il2, B2, C2, D2, ctx.Q)

        grads = naju_chunk_backward(u, f_logit, i_logit, B, C, D_skip,
                                    dy, Q=ctx.Q, SB=16, fwd_fn=_fwd)
        return (*grads, None)


_FLOGIT_SAFE_MIN = -5.0   # shared backward (SB=16 decay) envelope; fwd needs none


def naju_affine_chunk_scan(u, f_logit, i_logit, B, C, D_skip, Q=None,
                           fallback=None, check_range=True):
    """Hierarchical affine chunk scan. Forward exact for any f_logit; when
    grads are required the shared fused backward keeps the chunk backend's
    f_logit >= -5 envelope, so the same guard/fallback applies (training only).
    """
    if Q is None:
        Q = int(os.environ.get("NAJU_CHUNK_SIZE", 64))
    needs_grad = torch.is_grad_enabled() and any(
        t.requires_grad for t in (u, f_logit, i_logit, B, C))
    if check_range and needs_grad and float(f_logit.detach().min()) < _FLOGIT_SAFE_MIN:
        if fallback is None:
            from naju.cuda import naju_cuda_scan as fallback
        return fallback(u, f_logit, i_logit, B, C, D_skip)
    T = u.shape[1]
    pad = (Q - T % Q) % Q
    if pad:
        # neutral tail (decay ~1, write ~0), causal — cannot affect y[:T]
        u = F.pad(u, (0, 0, 0, pad))
        f_logit = F.pad(f_logit, (0, 0, 0, pad), value=40.0)
        i_logit = F.pad(i_logit, (0, 0, 0, pad), value=-40.0)
        B = F.pad(B, (0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, pad))
    y = _AffineChunkScan.apply(u, f_logit, i_logit, B, C, D_skip, Q)
    return y[:, :T] if pad else y
