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

Backward: exact for any f_logit, hybrid dispatch, no fallback backend.
Inside the decay-ratio envelope (f_logit >= -5, all trained-gate statistics
we observe) it uses the shared v4 fused backward with THIS forward injected
for the time-reversed dw pass. Outside the envelope — where the v4 math
NaNs and the chunk backend must fall back to the slow cuda scan — it runs
an affine-native path: the adjoint recurrence xbar_t = dy_t (x) C_t +
f_{t+1} xbar_{t+1} is itself affine, so SB-block boundary states for both
directions come from the Phase A summary kernel + one exclusive scan each,
and a per-(batch, SB-block) kernel recomputes forward states from the block
boundaries (register-resident re-walk, no [B,T,D,N] materialization) and
emits dfl, du/dil (via <xbar, B>), and the cross-channel dB/dC reductions.
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
                         Q: tl.constexpr, SB: tl.constexpr,
                         BD: tl.constexpr, BN: tl.constexpr,
                         EMIT_CZ: tl.constexpr):
    """Phase A: local scan z_q = f_q*z_{q-1} + i_q*u_q*B_q from zero, state
    [BD, BN] in registers. Inputs are loaded and outputs stored as SB-token
    2D blocks (rows extracted/scattered in registers): ~2x faster than
    per-token vector loads, which leave the sequential chain latency-bound."""
    pid_bk = tl.program_id(0)          # fused (batch, chunk)
    pid_d = tl.program_id(1)           # channel tile
    K = T // Q
    b = pid_bk // K
    k = pid_bk % K

    offs_s = tl.arange(0, SB)
    offs_d = pid_d * BD + tl.arange(0, BD)
    offs_n = tl.arange(0, BN)
    dm = offs_d < D
    nm = offs_n < N
    m2 = dm[None, :]

    z = tl.zeros((BD, BN), dtype=tl.float32)
    p = tl.zeros((BD,), dtype=tl.float32) + 1.0
    NSB: tl.constexpr = Q // SB
    for sb in range(NSB):
        t0 = k * Q + sb * SB
        td2 = (b * T + t0 + offs_s)[:, None] * D + offs_d[None, :]
        fl_b = tl.load(fl_ptr + td2, mask=m2, other=0.0).to(tl.float32)
        il_b = tl.load(il_ptr + td2, mask=m2, other=0.0).to(tl.float32)
        u_b = tl.load(u_ptr + td2, mask=m2, other=0.0).to(tl.float32)
        bn2 = (b * T + t0 + offs_s)[:, None] * N + offs_n[None, :]
        B_b = tl.load(B_ptr + bn2, mask=nm[None, :], other=0.0).to(tl.float32)
        C_b = tl.load(C_ptr + bn2, mask=nm[None, :], other=0.0).to(tl.float32)

        f_b = tl.sigmoid(fl_b)
        w_b = tl.sigmoid(il_b) * u_b                  # [SB, BD]
        cz_b = tl.zeros((SB, BD), dtype=tl.float32)
        p_b = tl.zeros((SB, BD), dtype=tl.float32)
        for q in tl.static_range(SB):
            eq = offs_s == q
            fq = tl.sum(tl.where(eq[:, None], f_b, 0.0), axis=0)   # [BD]
            wq = tl.sum(tl.where(eq[:, None], w_b, 0.0), axis=0)
            Bq = tl.sum(tl.where(eq[:, None], B_b, 0.0), axis=0)   # [BN]
            z = fq[:, None] * z + wq[:, None] * Bq[None, :]
            p = p * fq
            if EMIT_CZ:
                Cq = tl.sum(tl.where(eq[:, None], C_b, 0.0), axis=0)
                czq = tl.sum(Cq[None, :] * z, axis=1)              # [BD]
                cz_b = tl.where(eq[:, None], czq[None, :], cz_b)
                p_b = tl.where(eq[:, None], p[None, :], p_b)
        if EMIT_CZ:
            tl.store(cz_ptr + td2, cz_b.to(cz_ptr.dtype.element_ty), mask=m2)
            tl.store(P_ptr + td2, p_b.to(P_ptr.dtype.element_ty), mask=m2)

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


@triton.jit
def _affine_bwd_kernel(u_ptr, fl_ptr, il_ptr, B_ptr, C_ptr, dy_ptr,
                       xin_ptr, rbar_ptr,
                       dfl_ptr, dws_ptr, dB_ptr, dC_ptr,
                       T, D, N,
                       SB: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr):
    """Per-(batch, SB-block) gradients by chunk recompute — no decay ratios,
    exact for any f_logit.

    Forward state x rebuilt by re-walking from the block-start state xin
    (O(SB^2/2) extra state FMAs, register-resident); adjoint xbar walked in
    reverse from the block-end carry rbar = f_{t0+SB} * xbar_{t0+SB}.
    Emits per token: dfl, dws = <xbar, B_t> (-> du/dil outside), and the
    cross-channel reductions dB, dC accumulated over the in-program D loop.
    """
    pid = tl.program_id(0)              # fused (batch, SB-block)
    Ks = T // SB
    b = pid // Ks
    ks = pid % Ks
    t0 = b * T + ks * SB

    offs_s = tl.arange(0, SB)
    offs_n = tl.arange(0, BN)
    nm = offs_n < N

    bn2 = (t0 + offs_s)[:, None] * N + offs_n[None, :]
    B_b = tl.load(B_ptr + bn2, mask=nm[None, :], other=0.0).to(tl.float32)
    C_b = tl.load(C_ptr + bn2, mask=nm[None, :], other=0.0).to(tl.float32)

    dB_b = tl.zeros((SB, BN), dtype=tl.float32)
    dC_b = tl.zeros((SB, BN), dtype=tl.float32)

    for dt in range(0, tl.cdiv(D, BD)):
        offs_d = dt * BD + tl.arange(0, BD)
        dm = offs_d < D
        m2 = dm[None, :]
        td2 = (t0 + offs_s)[:, None] * D + offs_d[None, :]
        fl_b = tl.load(fl_ptr + td2, mask=m2, other=0.0).to(tl.float32)
        il_b = tl.load(il_ptr + td2, mask=m2, other=0.0).to(tl.float32)
        u_b = tl.load(u_ptr + td2, mask=m2, other=0.0).to(tl.float32)
        dy_b = tl.load(dy_ptr + td2, mask=m2, other=0.0).to(tl.float32)
        f_b = tl.sigmoid(fl_b)
        w_b = tl.sigmoid(il_b) * u_b                    # [SB, BD]

        sn = ((b * Ks + ks) * D + offs_d)[:, None] * N + offs_n[None, :]
        ld = dm[:, None] & nm[None, :]
        xin = tl.load(xin_ptr + sn, mask=ld, other=0.0)      # [BD, BN] fp32
        carry = tl.load(rbar_ptr + sn, mask=ld, other=0.0)   # f_{end}*xbar_{end}

        dfl_b = tl.zeros((SB, BD), dtype=tl.float32)
        dws_b = tl.zeros((SB, BD), dtype=tl.float32)
        for i in tl.static_range(SB):
            t = SB - 1 - i
            eq = offs_s == t
            f_t = tl.sum(tl.where(eq[:, None], f_b, 0.0), axis=0)   # [BD]
            w_t = tl.sum(tl.where(eq[:, None], w_b, 0.0), axis=0)
            dy_t = tl.sum(tl.where(eq[:, None], dy_b, 0.0), axis=0)
            B_t = tl.sum(tl.where(eq[:, None], B_b, 0.0), axis=0)   # [BN]
            C_t = tl.sum(tl.where(eq[:, None], C_b, 0.0), axis=0)

            xbar = dy_t[:, None] * C_t[None, :] + carry             # [BD, BN]
            carry = f_t[:, None] * xbar                             # for t-1

            # rebuild x_{t-1} from the block-start state (t steps)
            xprev = xin
            for j in range(0, t):
                ej = offs_s == j
                f_j = tl.sum(tl.where(ej[:, None], f_b, 0.0), axis=0)
                w_j = tl.sum(tl.where(ej[:, None], w_b, 0.0), axis=0)
                B_j = tl.sum(tl.where(ej[:, None], B_b, 0.0), axis=0)
                xprev = f_j[:, None] * xprev + w_j[:, None] * B_j[None, :]
            x_t = f_t[:, None] * xprev + w_t[:, None] * B_t[None, :]

            dot_fx = tl.sum(xbar * xprev, axis=1)                   # [BD]
            dfl_t = dot_fx * f_t * (1.0 - f_t)
            dws_t = tl.sum(xbar * B_t[None, :], axis=1)             # [BD]
            dfl_b = tl.where(eq[:, None], dfl_t[None, :], dfl_b)
            dws_b = tl.where(eq[:, None], dws_t[None, :], dws_b)

            dB_b += tl.where(eq[:, None],
                             tl.sum(xbar * w_t[:, None], axis=0)[None, :], 0.0)
            dC_b += tl.where(eq[:, None],
                             tl.sum(dy_t[:, None] * x_t, axis=0)[None, :], 0.0)

        tl.store(dfl_ptr + td2, dfl_b, mask=m2)
        tl.store(dws_ptr + td2, dws_b, mask=m2)

    tl.store(dB_ptr + bn2, dB_b, mask=nm[None, :])
    tl.store(dC_ptr + bn2, dC_b, mask=nm[None, :])


def _sb_boundary_states(u, fl, il, B, SB=16):
    """State entering each SB block of the affine system
    x' = sigmoid(fl)*x + (sigmoid(il)*u) B^T: Phase A at Q=SB (summaries
    only) + one register-resident exclusive scan. Guard-free."""
    from naju.chunk_bwd_triton import _exclusive_scan
    B_, T, D = fl.shape
    N = B.shape[-1]
    Ks = T // SB
    F_s = torch.empty(B_, Ks, D, device=fl.device, dtype=torch.float32)
    W_s = torch.empty(B_, Ks, D, N, device=fl.device, dtype=torch.float32)
    BD = 64
    BN = max(16, triton.next_power_of_2(N))
    grid = (B_ * Ks, triton.cdiv(D, BD))
    _affine_local_kernel[grid](u, fl, il, B, B, F_s, F_s, F_s, W_s,
                               T, D, N, Q=SB, SB=SB, BD=BD, BN=BN,
                               EMIT_CZ=0, num_warps=2)
    return _exclusive_scan(F_s, W_s)


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
                                   T, D, N, Q=Q, SB=16, BD=BD, BN=BN,
                                   EMIT_CZ=1, num_warps=2)

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
        u, f_logit, i_logit, B, C, D_skip = ctx.saved_tensors

        # hybrid dispatch: inside the decay-ratio envelope (all trained-gate
        # statistics we observe) the shared v4 fused backward is ~2x faster;
        # outside it, it is wrong (NaN) — use the exact affine-native path.
        if float(f_logit.detach().min()) >= _FLOGIT_SAFE_MIN:
            from naju.chunk_bwd_triton import naju_chunk_backward

            def _fwd(u2, fl2, il2, B2, C2, D2):
                return _AffineChunkScan.apply(u2, fl2, il2, B2, C2, D2, ctx.Q)

            grads = naju_chunk_backward(u, f_logit, i_logit, B, C, D_skip,
                                        dy, Q=ctx.Q, SB=16, fwd_fn=_fwd)
            return (*grads, None)

        B_, T, D = u.shape
        N = B.shape[-1]
        dt = torch.float32
        uc, flc, ilc, dyc = (t.contiguous().to(dt)
                             for t in (u, f_logit, i_logit, dy))
        Bc, Cc = B.contiguous().to(dt), C.contiguous().to(dt)
        SB = 16
        Ks = T // SB

        # x entering each SB block, and the adjoint carry leaving it:
        # xbar_t = dy_t (x) C_t + f_{t+1} xbar_{t+1}; the carry
        # c_t = f_{t+1} xbar_{t+1} is itself affine (decay f, drive f*dy (x) C)
        # scanned on the flipped time axis — same Phase A machinery.
        xin = _sb_boundary_states(uc, flc, ilc, Bc, SB)
        S = torch.sigmoid(flc) * dyc
        pos = torch.full_like(flc, 40.0)                 # sigmoid ~ 1
        rbar = torch.flip(
            _sb_boundary_states(torch.flip(S, [1]), torch.flip(flc, [1]),
                                pos, torch.flip(Cc, [1]), SB), [1])

        dfl = torch.empty(B_, T, D, device=u.device, dtype=dt)
        dws = torch.empty(B_, T, D, device=u.device, dtype=dt)
        dB = torch.empty(B_, T, N, device=u.device, dtype=dt)
        dC = torch.empty(B_, T, N, device=u.device, dtype=dt)
        BN = max(16, triton.next_power_of_2(N))
        _affine_bwd_kernel[(B_ * Ks,)](uc, flc, ilc, Bc, Cc, dyc,
                                       xin, rbar, dfl, dws, dB, dC,
                                       T, D, N, SB=SB, BD=16, BN=BN,
                                       num_warps=2)

        i_g = torch.sigmoid(ilc)
        du = i_g * dws + D_skip.to(dt).unsqueeze(0) * dyc
        dil = dws * uc * i_g * (1.0 - i_g)
        dD = (dyc * uc).sum(dim=(0, 1))
        outs = (du, dfl, dil, dB, dC, dD)
        return (*(o.to(u.dtype) for o in outs), None)


_FLOGIT_SAFE_MIN = -5.0   # v4 backward fast-path envelope (see backward())


def naju_affine_chunk_scan(u, f_logit, i_logit, B, C, D_skip, Q=None,
                           fallback=None, check_range=True):
    """Hierarchical affine chunk scan, exact for any f_logit in BOTH forward
    and backward — no fallback backend. Backward picks the shared v4 fused
    kernels inside their f_logit >= -5 envelope (faster) and the exact
    affine-native path outside it. `fallback`/`check_range` are accepted for
    signature compatibility and ignored.
    """
    if Q is None:
        Q = int(os.environ.get("NAJU_CHUNK_SIZE", 64))
    if Q % 16 != 0:
        raise ValueError(f"Q={Q} must be a multiple of 16 (SB-block loads)")
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
