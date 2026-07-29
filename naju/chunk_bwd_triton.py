"""Fused Triton backward for the chunk scan: dB/dC/dfl in one kernel.

Split of labor:
  dw          : time-reversed call of the fused Triton FORWARD (exact).
  dB/dC/dfl   : THIS kernel. One program per (batch, SB-block); inner loop over
                channel tiles accumulates the cross-channel reductions (M2 Gram,
                dB/dC boundary terms) and emits dfl per tile. All cumsums are
                SB-local (span 16 -> fp32-safe for f_logit >= ~-5.5, same guard
                as the forward).
  boundaries  : x_in / R at SB granularity stay in torch (K + NSB batched steps,
                cheap); they are inputs to the kernel.
  du/dil/dD   : elementwise torch (negligible).
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _bwd_kernel(fl_ptr, il_ptr, u_ptr, dy_ptr, dw_ptr, B_ptr, C_ptr,
                xin_ptr, R_ptr, dB_ptr, dC_ptr, dfl_ptr,
                T, D, N,
                SB: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)              # fused (batch, sub-block) index
    Ks = T // SB
    b = pid // Ks
    ks = pid % Ks
    t0 = b * T + ks * SB

    offs_s = tl.arange(0, SB)
    offs_n = tl.arange(0, BN)
    n_mask = offs_n < N

    bc_off = (t0 + offs_s)[:, None] * N + offs_n[None, :]
    Bs = tl.load(B_ptr + bc_off, mask=n_mask[None, :], other=0.0).to(tl.float32)
    Cs = tl.load(C_ptr + bc_off, mask=n_mask[None, :], other=0.0).to(tl.float32)

    causal = offs_s[:, None] >= offs_s[None, :]          # [s, t] : s >= t
    G = tl.dot(Cs, tl.trans(Bs), allow_tf32=False)
    Gm = tl.where(causal, G, 0.0)

    M2 = tl.zeros((SB, SB), dtype=tl.float32)
    dBa = tl.zeros((SB, BN), dtype=tl.float32)
    dCa = tl.zeros((SB, BN), dtype=tl.float32)

    for dt in range(0, tl.cdiv(D, BD)):
        offs_d = dt * BD + tl.arange(0, BD)
        d_mask = offs_d < D
        m2 = d_mask[None, :]
        td_off = (t0 + offs_s)[:, None] * D + offs_d[None, :]

        fl = tl.load(fl_ptr + td_off, mask=m2, other=0.0).to(tl.float32)
        il = tl.load(il_ptr + td_off, mask=m2, other=0.0).to(tl.float32)
        uu = tl.load(u_ptr + td_off, mask=m2, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + td_off, mask=m2, other=0.0).to(tl.float32)
        dw = tl.load(dw_ptr + td_off, mask=m2, other=0.0).to(tl.float32)

        f = tl.sigmoid(fl)
        lf = -tl.log(1.0 + tl.exp(-fl))
        L = tl.cumsum(lf, axis=0)                        # inclusive, span <= SB
        E = tl.exp(L)
        Einv = tl.exp(-L)
        Pend = tl.exp(tl.sum(lf, axis=0))                # [BD]
        w = tl.sigmoid(il) * uu
        wE = w * Einv
        dyE = dy * E

        sn_off = ((b * Ks + ks) * D + offs_d)[:, None] * N + offs_n[None, :]
        ld = d_mask[:, None] & n_mask[None, :]
        xin = tl.load(xin_ptr + sn_off, mask=ld, other=0.0).to(tl.float32)
        Rr = tl.load(R_ptr + sn_off, mask=ld, other=0.0).to(tl.float32)

        # cross-channel accumulators ----------------------------------
        M2 += tl.dot(dyE, tl.trans(wE), allow_tf32=False)
        dBa += tl.dot(wE * Pend[None, :], Rr, allow_tf32=False)
        dCa += tl.dot(dyE, xin, allow_tf32=False)

        # dfl for this tile --------------------------------------------
        dwcore = tl.dot(tl.trans(Gm), dyE, allow_tf32=False)   # [t,d] sum_{s>=t}
        introw = tl.dot(Gm, wE, allow_tf32=False)              # [t,d] sum_{i<=t}
        a = Pend * tl.sum(xin * Rr, axis=1)                    # [BD]
        cx = tl.dot(Cs, tl.trans(xin), allow_tf32=False)       # [SB, BD]
        tmp = dyE * cx
        bterm = tl.sum(tmp, axis=0)[None, :] - tl.cumsum(tmp, axis=0) + tmp
        bR = tl.dot(Bs, tl.trans(Rr), allow_tf32=False)
        cterm = Pend[None, :] * tl.cumsum(wE * bR, axis=0)
        t2 = dyE * introw
        dterm = tl.cumsum(wE * dwcore, axis=0) - (tl.cumsum(t2, axis=0) - t2)
        s_t = a[None, :] + bterm + cterm + dterm
        dfl = (1.0 - f) * (s_t - w * dw)
        tl.store(dfl_ptr + td_off, dfl, mask=m2)

    M2m = tl.where(causal, M2, 0.0)                      # s >= i
    dB = tl.dot(tl.trans(M2m), Cs, allow_tf32=False) + dBa
    dC = tl.dot(M2m, Bs, allow_tf32=False) + dCa
    st_mask = n_mask[None, :]
    tl.store(dB_ptr + bc_off, dB, mask=st_mask)
    tl.store(dC_ptr + bc_off, dC, mask=st_mask)


@triton.jit
def _affine_scan_kernel(P_ptr, S_ptr, out_ptr, K, D, N, NT,
                        REV: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr):
    """Exclusive prefix of x <- P*x + S over the block axis, state in registers.
    out[k] = state entering step k (zeros at the starting end). REV runs the
    same recurrence from the last block down (for the reverse adjoint carry).
    N is tiled (grid axis 2) to keep occupancy up when B*D/BD is small."""
    b = tl.program_id(0)
    pd = tl.program_id(1)
    pn = tl.program_id(2)
    offs_d = pd * BD + tl.arange(0, BD)
    offs_n = pn * BN + tl.arange(0, BN)
    dm = offs_d < D
    m = dm[:, None] & (offs_n < N)[None, :]
    x = tl.zeros((BD, BN), dtype=tl.float32)
    for kk in range(0, K):
        if REV:
            k = K - 1 - kk
        else:
            k = kk
        base = b * K + k
        off2 = (base * D + offs_d)[:, None] * N + offs_n[None, :]
        tl.store(out_ptr + off2, x, mask=m)
        P = tl.load(P_ptr + base * D + offs_d, mask=dm, other=1.0).to(tl.float32)
        S = tl.load(S_ptr + off2, mask=m, other=0.0).to(tl.float32)
        x = P[:, None] * x + S


def _exclusive_scan(P, S, rev=False):
    B_, K, D, N = S.shape
    Pc, Sc = P.contiguous(), S.contiguous()
    out = torch.empty_like(Sc)
    BD = 64
    BN = 16
    grid = (B_, triton.cdiv(D, BD), triton.cdiv(N, BN))
    _affine_scan_kernel[grid](Pc, Sc, out, K, D, N, grid[2],
                              REV=rev, BD=BD, BN=BN)
    return out


@triton.jit
def _sb_summary_kernel(u_ptr, fl_ptr, il_ptr, dy_ptr, B_ptr, C_ptr,
                       Sk_ptr, St_ptr, Pend_ptr, T, D, N,
                       SB: tl.constexpr, BD: tl.constexpr, BN: tl.constexpr):
    """Per SB-block summaries straight from raw inputs (no [B,T,D] torch
    elementwise or einsum copies):  Pend = prod(f),  Sk[d,n] = Pend_d *
    sum_q (wE)_{q,d} B_{q,n},  Stil[d,n] = sum_q (dy*E)_{q,d} C_{q,n}."""
    pid = tl.program_id(0)              # fused (batch, sb-block)
    pd = tl.program_id(1)
    Ks = T // SB
    b = pid // Ks
    ks = pid % Ks
    t0 = b * T + ks * SB

    offs_s = tl.arange(0, SB)
    offs_d = pd * BD + tl.arange(0, BD)
    offs_n = tl.arange(0, BN)
    dm = offs_d < D
    nm = offs_n < N
    m2 = dm[None, :]

    td_off = (t0 + offs_s)[:, None] * D + offs_d[None, :]
    fl = tl.load(fl_ptr + td_off, mask=m2, other=0.0).to(tl.float32)
    il = tl.load(il_ptr + td_off, mask=m2, other=0.0).to(tl.float32)
    uu = tl.load(u_ptr + td_off, mask=m2, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + td_off, mask=m2, other=0.0).to(tl.float32)
    bc_off = (t0 + offs_s)[:, None] * N + offs_n[None, :]
    Bs = tl.load(B_ptr + bc_off, mask=nm[None, :], other=0.0).to(tl.float32)
    Cs = tl.load(C_ptr + bc_off, mask=nm[None, :], other=0.0).to(tl.float32)

    lf = -tl.log(1.0 + tl.exp(-fl))
    L = tl.cumsum(lf, axis=0)
    Pend = tl.exp(tl.sum(lf, axis=0))                    # [BD]
    wE = tl.sigmoid(il) * uu * tl.exp(-L)
    dyE = dy * tl.exp(L)

    Sk = tl.dot(tl.trans(wE * Pend[None, :]), Bs, allow_tf32=False)   # [BD,BN]
    St = tl.dot(tl.trans(dyE), Cs, allow_tf32=False)
    sn_off = ((b * Ks + ks) * D + offs_d)[:, None] * N + offs_n[None, :]
    sm = dm[:, None] & nm[None, :]
    tl.store(Sk_ptr + sn_off, Sk, mask=sm)
    tl.store(St_ptr + sn_off, St, mask=sm)
    tl.store(Pend_ptr + (b * Ks + ks) * D + offs_d, Pend, mask=dm)


def naju_chunk_backward(u, f_logit, i_logit, B, C, D_skip, dy,
                           Q=64, SB=16, fwd_fn=None):
    B_, T, D = u.shape
    N = B.shape[-1]
    assert T % Q == 0 and Q % SB == 0 and SB == 16
    K, NSB, Ks = T // Q, Q // SB, T // SB
    dev = u.device
    dt = torch.float32
    uc, flc, ilc, dyc = (t.contiguous().to(dt) for t in (u, f_logit, i_logit, dy))
    Bc, Cc = B.contiguous().to(dt), C.contiguous().to(dt)

    # ---- dw via time-reversed fused forward --------------------------
    if fwd_fn is None:
        # the caller must supply the forward scan (chunk_triton passes its
        # own fused forward)
        raise ValueError("naju_chunk_backward requires fwd_fn")
    fl_rev = torch.cat([torch.full_like(flc[:, :1], 40.0),
                        torch.flip(flc, [1])[:, :-1]], dim=1)
    with torch.no_grad():
        yprime = fwd_fn(torch.flip(dyc, [1]), fl_rev,
                        torch.full_like(ilc, 40.0),
                        torch.flip(Cc, [1]), torch.flip(Bc, [1]),
                        torch.zeros(D, dtype=dt, device=dev))
    dw = torch.flip(yprime.to(dt), [1]).contiguous()

    i_g = torch.sigmoid(ilc)
    du = i_g * dw + D_skip.to(dt).unsqueeze(0) * dyc
    dil = dw * uc * i_g * (1.0 - i_g)
    dD = (dyc * uc).sum(dim=(0, 1))

    # ---- boundary states at SB granularity (two small kernels) -------
    # SB summaries straight from raw inputs, then a single register-resident
    # affine scan over all Ks blocks (no chunk-level hierarchy needed).
    Sk_s = torch.empty(B_, Ks, D, N, device=dev, dtype=dt)
    St_s = torch.empty(B_, Ks, D, N, device=dev, dtype=dt)
    Pend_s = torch.empty(B_, Ks, D, device=dev, dtype=dt)
    BD = 64
    BN = max(16, triton.next_power_of_2(N))
    _sb_summary_kernel[(B_ * Ks, triton.cdiv(D, BD))](
        uc, flc, ilc, dyc, Bc, Cc, Sk_s, St_s, Pend_s,
        T, D, N, SB=SB, BD=BD, BN=BN, num_stages=1)
    xin_s = _exclusive_scan(Pend_s, Sk_s)
    R_s = _exclusive_scan(Pend_s, St_s, rev=True)

    # ---- fused kernel: dB / dC / dfl ----------------------------------
    dB = torch.empty(B_, T, N, device=dev, dtype=dt)
    dC = torch.empty(B_, T, N, device=dev, dtype=dt)
    dfl = torch.empty(B_, T, D, device=dev, dtype=dt)
    BD = 64
    BN = max(16, triton.next_power_of_2(N))
    grid = (B_ * Ks,)
    _bwd_kernel[grid](flc, ilc, uc, dyc, dw, Bc, Cc, xin_s, R_s,
                      dB, dC, dfl, T, D, N, SB=SB, BD=BD, BN=BN,
                      num_stages=1)

    outs = (du, dfl, dil, dB, dC, dD)
    return tuple(o.to(u.dtype) for o in outs)
