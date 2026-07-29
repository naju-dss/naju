"""Fused Triton chunk-parallel forward for the Naju v1 recurrence.

Three-pass SSD-style structure:
  pass 1 (Triton, fused): per (batch, chunk, channel-tile) program computes
          - Gram G = C @ B^T with causal mask (shared across channels),
          - within-chunk decay via logsigmoid + cumsum,
          - intra-chunk output  intra = exp(L) * (A @ (w * exp(-L))),
          - chunk-end partial state S_k = (w * exp(Ltot - L))^T @ B.
  pass 2 (torch, tiny):  sequential carry over K = T/Q chunk states.
  pass 3 (torch, bmm):   carry output  exp(L) * (C @ x_start^T)  via cuBLAS.

Backward: the v4 fused Triton backward (chunk_bwd_triton.naju_chunk_backward),
with dw computed via a time-reversed call of this same fused forward.

Numerics: single-level decay (no sub-chunking yet) — safe when the
within-chunk log-decay span (Q-1)*|log f_min| stays under ~80 in fp32:
Q=16 covers f_logit >= -5.5, Q=64 requires f_logit >= -1.4.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _pass1_kernel(u_ptr, fl_ptr, il_ptr, B_ptr, C_ptr,
                  intra_ptr, S_ptr, Pend_ptr,
                  T, D, N,
                  Q: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr):
    pid_bk = tl.program_id(0)          # fused (batch, chunk) index
    pid_d = tl.program_id(1)           # channel tile
    K = T // Q
    b = pid_bk // K
    k = pid_bk % K

    offs_q = tl.arange(0, Q)
    offs_d = pid_d * BD + tl.arange(0, BD)
    offs_n = tl.arange(0, BN)
    d_mask = offs_d < D
    n_mask = offs_n < N

    # ---- B/C for this chunk: [Q, BN] ----
    bc_off = (b * T + k * Q + offs_q)[:, None] * N + offs_n[None, :]
    Bc = tl.load(B_ptr + bc_off, mask=n_mask[None, :], other=0.0).to(tl.float32)
    Cc = tl.load(C_ptr + bc_off, mask=n_mask[None, :], other=0.0).to(tl.float32)

    # ---- Gram with causal mask (i <= t), diagonal included ----
    G = tl.dot(Cc, tl.trans(Bc), allow_tf32=False)                       # [Q, Q]
    causal = offs_q[:, None] >= offs_q[None, :]
    A = tl.where(causal, G, 0.0)

    # ---- gates / decay for this channel tile: [Q, BD] ----
    td_off = (b * T + k * Q + offs_q)[:, None] * D + offs_d[None, :]
    m2 = d_mask[None, :]
    fl = tl.load(fl_ptr + td_off, mask=m2, other=0.0).to(tl.float32)
    il = tl.load(il_ptr + td_off, mask=m2, other=0.0).to(tl.float32)
    uu = tl.load(u_ptr + td_off, mask=m2, other=0.0).to(tl.float32)

    lf = -tl.log(1.0 + tl.exp(-fl))                    # logsigmoid(fl)
    L = tl.cumsum(lf, axis=0)                          # inclusive
    w = tl.sigmoid(il) * uu
    Wt = w * tl.exp(-L)

    intra = tl.exp(L) * tl.dot(A, Wt, allow_tf32=False)                  # [Q, BD]
    tl.store(intra_ptr + td_off, intra, mask=m2)

    # ---- chunk-end partial state and total decay ----
    Ltot = tl.sum(lf, axis=0)                          # [BD]
    Pend = tl.exp(Ltot)
    pend_off = (b * K + k) * D + offs_d
    tl.store(Pend_ptr + pend_off, Pend, mask=d_mask)

    Sk = tl.dot(tl.trans(Wt * Pend[None, :]), Bc, allow_tf32=False)      # [BD, BN]
    s_off = ((b * K + k) * D + offs_d)[:, None] * N + offs_n[None, :]
    tl.store(S_ptr + s_off, Sk, mask=d_mask[:, None] & n_mask[None, :])



@triton.jit
def _pass1_kernel_v2(u_ptr, fl_ptr, il_ptr, B_ptr, C_ptr,
                     intra_ptr, S_ptr, Pend_ptr, eL_ptr,
                     T, D, N,
                     Q: tl.constexpr, SB: tl.constexpr,
                     BD: tl.constexpr, BN: tl.constexpr,
                     TF32: tl.constexpr):
    """Two-level chunking: chunk Q processed as Q//SB sequential sub-blocks
    with an in-register running state, so every exp() argument is bounded by
    the SB-step decay span (SB=16 -> safe to f_logit ~ -5.5; SB=8 -> ~ -12)."""
    pid_bk = tl.program_id(0)
    pid_d = tl.program_id(1)
    K = T // Q
    b = pid_bk // K
    k = pid_bk % K

    offs_s = tl.arange(0, 16)
    offs_d = pid_d * BD + tl.arange(0, BD)
    offs_n = tl.arange(0, BN)
    d_mask = offs_d < D
    n_mask = offs_n < N
    m2 = d_mask[None, :]

    # physical tile is 16 rows (tl.dot minimum); logical sub-block SB <= 16
    # rows beyond SB are masked out (their lf forced to 0, w to 0).
    xloc = tl.zeros((BD, BN), dtype=tl.float32)   # state since chunk start
    Ptot = tl.zeros((BD,), dtype=tl.float32) + 1.0
    NSB: tl.constexpr = Q // SB
    row_ok = offs_s < SB
    for sb in tl.static_range(NSB):
        t0 = k * Q + sb * SB
        bc_off = (b * T + t0 + offs_s)[:, None] * N + offs_n[None, :]
        ld_mask = row_ok[:, None] & n_mask[None, :]
        Bs = tl.load(B_ptr + bc_off, mask=ld_mask, other=0.0).to(tl.float32)
        Cs = tl.load(C_ptr + bc_off, mask=ld_mask, other=0.0).to(tl.float32)

        td_off = (b * T + t0 + offs_s)[:, None] * D + offs_d[None, :]
        m2r = m2 & row_ok[:, None]
        fl = tl.load(fl_ptr + td_off, mask=m2r, other=0.0).to(tl.float32)
        il = tl.load(il_ptr + td_off, mask=m2r, other=0.0).to(tl.float32)
        uu = tl.load(u_ptr + td_off, mask=m2r, other=0.0).to(tl.float32)

        lf = -tl.log(1.0 + tl.exp(-fl))
        lf = tl.where(row_ok[:, None], lf, 0.0)    # pad rows: neutral decay
        Ls = tl.cumsum(lf, axis=0)                 # span <= SB steps
        eLs = tl.exp(Ls)
        Wt = tl.sigmoid(il) * uu * tl.exp(-Ls)     # pad rows: uu=0 -> Wt=0

        G = tl.dot(Cs, tl.trans(Bs), allow_tf32=TF32)
        causal = offs_s[:, None] >= offs_s[None, :]
        A = tl.where(causal, G, 0.0)

        y_s = eLs * tl.dot(A, Wt, allow_tf32=TF32)
        y_s += eLs * tl.dot(Cs, tl.trans(xloc), allow_tf32=TF32)
        tl.store(intra_ptr + td_off, y_s, mask=m2r)
        tl.store(eL_ptr + td_off, eLs * Ptot[None, :], mask=m2r)

        Pend_s = tl.exp(tl.sum(lf, axis=0))        # [BD]
        Sk_s = tl.dot(tl.trans(Wt * Pend_s[None, :]), Bs, allow_tf32=TF32)
        xloc = Pend_s[:, None] * xloc + Sk_s
        Ptot = Ptot * Pend_s

    pend_off = (b * K + k) * D + offs_d
    tl.store(Pend_ptr + pend_off, Ptot, mask=d_mask)
    s_off = ((b * K + k) * D + offs_d)[:, None] * N + offs_n[None, :]
    tl.store(S_ptr + s_off, xloc, mask=d_mask[:, None] & n_mask[None, :])


class _ChunkScanTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, f_logit, i_logit, B, C, D_skip, Q, SB, tf32):
        B_, T, D = u.shape
        N = B.shape[-1]
        K = T // Q
        dt = torch.float32
        uc, flc, ilc = (t.contiguous().to(dt) for t in (u, f_logit, i_logit))
        Bc_, Cc_ = B.contiguous().to(dt), C.contiguous().to(dt)

        intra = torch.empty(B_, T, D, device=u.device, dtype=dt)
        S = torch.empty(B_, K, D, N, device=u.device, dtype=dt)
        Pend = torch.empty(B_, K, D, device=u.device, dtype=dt)
        eL = torch.empty(B_, T, D, device=u.device, dtype=dt)
        BD = 64
        BN = max(16, triton.next_power_of_2(N))
        grid = (B_ * K, triton.cdiv(D, BD))
        _pass1_kernel_v2[grid](uc, flc, ilc, Bc_, Cc_, intra, S, Pend, eL,
                               T, D, N, Q=Q, SB=SB, BD=BD, BN=BN, TF32=tf32)

        # pass 2: sequential inter-chunk carry (K steps, batched)
        x_start = torch.zeros(B_, K, D, N, device=u.device, dtype=dt)
        carry = torch.zeros(B_, D, N, device=u.device, dtype=dt)
        for k in range(K):
            x_start[:, k] = carry
            carry = Pend[:, k].unsqueeze(-1) * carry + S[:, k]

        # pass 3: carry output via bmm (eL from the kernel; chunk-relative)
        Ck = Cc_.view(B_, K, Q, N)
        carry_y = eL.view(B_, K, Q, D) * torch.einsum("bkqn,bkdn->bkqd", Ck, x_start)

        y = intra + carry_y.reshape(B_, T, D) + D_skip.to(dt).unsqueeze(0) * uc
        ctx.save_for_backward(u, f_logit, i_logit, B, C, D_skip)
        ctx.Q = Q
        return y.to(u.dtype)

    @staticmethod
    def backward(ctx, dy):
        # v4 fused backward (verified vs the v3 reference at fp32 precision,
        # v3 itself vs autograd at machine precision): dw via a time-reversed
        # call of THIS fused forward; dB/dC/dfl in one Triton kernel; boundary
        # states via the SB-summary + register-resident affine-scan kernels.
        from naju.chunk_bwd_triton import naju_chunk_backward
        u, f_logit, i_logit, B, C, D_skip = ctx.saved_tensors

        def _fwd(u2, fl2, il2, B2, C2, D2):
            return _ChunkScanTriton.apply(u2, fl2, il2, B2, C2, D2, ctx.Q, 16, False)

        grads = naju_chunk_backward(u, f_logit, i_logit, B, C, D_skip,
                                       dy, Q=ctx.Q, SB=16, fwd_fn=_fwd)
        return (*grads, None, None, None)


_FLOGIT_SAFE_MIN = -5.0   # fp32 pairwise-decay budget at SB=16 (span 15*5.5 < 88)


def naju_chunk_scan(u, f_logit, i_logit, B, C, D_skip, Q=64, SB=16,
                             fallback=None, check_range=True, tf32=None):
    """Fused chunked scan with a numerical-safety guard.

    The single-shift fp32 decay math is exact for f_logit >= ~-5.5 at SB=16
    (covers all trained-gate statistics we observe). If the batch contains
    deeper gate logits, we fall back to the sequential CUDA scan (exact for
    any logit). Handling deep gates inside the kernel (per-channel pairwise
    tiles / FLA-style secondary rescaling) is future work.
    """
    if tf32 is None:
        import os
        tf32 = os.environ.get("NAJU_CHUNK_TF32", "0") == "1"
    if check_range and float(f_logit.detach().min()) < _FLOGIT_SAFE_MIN:
        if fallback is None:
            from naju.cuda import naju_cuda_scan as fallback
        return fallback(u, f_logit, i_logit, B, C, D_skip)
    T = u.shape[1]
    pad = (Q - T % Q) % Q
    if pad:
        # neutral tail: decay ~1 (fl=+40), write ~0 (il=-40, u=0) — causal, so
        # padded steps cannot affect y[:T]; the slice backward zero-pads dy.
        u = F.pad(u, (0, 0, 0, pad))
        f_logit = F.pad(f_logit, (0, 0, 0, pad), value=40.0)
        i_logit = F.pad(i_logit, (0, 0, 0, pad), value=-40.0)
        B = F.pad(B, (0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, pad))
    y = _ChunkScanTriton.apply(u, f_logit, i_logit, B, C, D_skip, Q, SB, tf32)
    return y[:, :T] if pad else y
