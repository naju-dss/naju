"""Selective SSM (Mamba S6) baseline mixer.

Standard selective state space model recurrence:

    deltaA_t = exp(delta_t * A)          (diagonal, per channel/state)
    deltaB_u_t = delta_t * B_t * u_t
    x_t = deltaA_t ⊙ x_{t-1} + deltaB_u_t
    y_t = sum_state(C_t ⊙ x_t) + D ⊙ u_t

The mamba-ssm fused CUDA kernel is used when available (selective_scan_fn);
otherwise the chunked pure-PyTorch scan is a drop-in fallback.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.scan import chunked_diagonal

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _selective_scan_fn
    _HAS_FUSED = True
except ImportError:
    _HAS_FUSED = False


def _auto_chunk(T, lo=16, hi=128):
    """Chunk size minimizing the number of sequential steps (~sqrt(T))."""
    c = int(round(T ** 0.5))
    return max(lo, min(hi, c))


class MambaMixer(nn.Module):
    """Standard selective SSM (Mamba S6) mixer."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", **_):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # A = -exp(A_log)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # dt bias init so softplus(dt) starts in a reasonable range
        with torch.no_grad():
            dt = torch.exp(
                torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
                + math.log(0.001)
            ).clamp(min=1e-4)
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, hidden):
        B, T, _ = hidden.shape
        xz = self.in_proj(hidden)
        x, z = xz.chunk(2, dim=-1)  # [B, T, d_inner]

        x = x.transpose(1, 2)
        x = self.conv1d(x)[..., :T]
        x = x.transpose(1, 2)
        x = F.silu(x)

        x_dbl = self.x_proj(x)
        dt, Bm, Cm = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        if _HAS_FUSED and x.is_cuda:
            # mamba-ssm fused CUDA kernel: handles softplus(delta + bias), D,
            # silu(z) all internally.
            dt_raw = self.dt_proj(dt)  # [B, T, d_inner] (bias included, pre-softplus)
            y = _selective_scan_fn(
                x.transpose(1, 2),           # [B, d_inner, T]
                dt_raw.transpose(1, 2),      # [B, d_inner, T]
                -torch.exp(self.A_log),      # [d_inner, d_state]
                Bm.transpose(1, 2),          # [B, d_state, T]
                Cm.transpose(1, 2),          # [B, d_state, T]
                self.D,                      # [d_inner]
                z=z.transpose(1, 2),         # [B, d_inner, T]
                delta_bias=self.dt_proj.bias,
                delta_softplus=True,
            )
            return self.out_proj(y.transpose(1, 2))

        # chunked pure-PyTorch scan fallback
        delta = F.softplus(self.dt_proj(dt))  # [B, T, d_inner]
        A = -torch.exp(self.A_log)             # [d_inner, d_state]
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # [B,T,d_inner,d_state]
        deltaB_u = delta.unsqueeze(-1) * Bm.unsqueeze(2) * x.unsqueeze(-1)
        xs = chunked_diagonal(deltaA, deltaB_u, chunk=_auto_chunk(T))
        y = (xs * Cm.unsqueeze(2)).sum(dim=-1)
        y = y + x * self.D
        y = y * F.silu(z)
        return self.out_proj(y)
