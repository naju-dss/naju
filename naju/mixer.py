"""Naju v1 mixer — the paper-final model.

Constructor defaults ARE the paper's final configuration, i.e. NajuMixer(d_model)
is exactly the model used in the paper's experiments:

    [x_raw, z] = W_in u                              (in_proj: d -> 2*d_inner)
    h_t = SiLU(DWConv(x_raw))                        (content branch, causal dwconv)
    [B_t, C_t] = W_x h_t                             (x_proj: d_inner -> 2N)
    a_t = W_f h_t + DWConv_f(h_t) + b_f              (forget logit, b_f init +5)
    c_t = W_i h_t + DWConv_i(h_t) + b_i              (input  logit, b_i init -2)
    x_t = sigmoid(a_t) x_{t-1} + sigmoid(c_t) B_t h_t    (scan, run with D=0)
    cx_t = (1/sqrt(N)) C_t . x_t                     (readout normalization)
    y_t = cx_t + D * h_t                             (feedthrough; D init 0.01)
    out = W_o(y_t * SiLU(z_t))                       (z-gate)

Notation follows the paper (u = residual stream, h = content branch; the code
variable `x` inside forward() is h). The mixer emits W_o(y*SiLU(z)) only — the
block-output residual u^{l+1} = u^l + out lives in NajuBlock, so the current
input reaches the next layer by two routes: the gated feedthrough D*h inside
the branch, and the clean identity path outside.

Options (all validated in the paper's experiments):
    d_init            D initialization (0.01 final; 1.0 = Mamba convention).
                      Small init lets the feedthrough start negligible and
                      grow only where the task wants it.
    gate_rank         low-rank W2@W1 factorization of the gate projections.
                      Ablation only: the gates are not the latency bottleneck
                      (the scan is), see the paper's diagnostic.
    gate_reparam      the full optimizer-matched gate reparameterization from
                      the paper's scale-up study: logit 1/sqrt(d_inner)
                      rescaling (weights scaled up by sqrt(d_inner) at init,
                      pre-activation divided back down — init-forward
                      identical) PLUS matched per-group lr/wd/Adam-eps, applied
                      automatically by optim.build_optimizer (it detects this
                      flag). The two halves are always used together (A/B
                      no-op verified across widths). Incompatible with
                      gate_rank.
    scan_backend      see scan.py (None = env / auto)

Parameter names match the original research code 1:1, so checkpoints trained
with it load directly (strict=True) into this class.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from naju.scan import naju_scan


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class NajuMixer(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, gate_conv_kernel=4,
                 forget_bias_init=5.0, input_bias_init=-2.0, d_init=0.01,
                 gate_rank=None, gate_reparam=False, scan_backend=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.gate_rank = gate_rank
        self.scan_backend = scan_backend
        self._cx_scale = 1.0 / (d_state ** 0.5)   # readout normalization (fixed)

        self.in_proj  = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                  groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.x_proj   = nn.Linear(self.d_inner, 2 * d_state, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.D = nn.Parameter(torch.full((self.d_inner,), float(d_init)))
        self.register_buffer("D_zero", torch.zeros(self.d_inner), persistent=False)

        def _gate_proj():
            if gate_rank:
                return nn.Sequential(
                    nn.Linear(self.d_inner, gate_rank, bias=False),
                    nn.Linear(gate_rank, self.d_inner, bias=False))
            return nn.Linear(self.d_inner, self.d_inner, bias=False)

        self.f_proj = _gate_proj()
        self.f_conv = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=gate_conv_kernel,
                                groups=self.d_inner, padding=gate_conv_kernel - 1, bias=False)
        self.f_bias = nn.Parameter(torch.full((self.d_inner,), float(forget_bias_init)))
        self.i_proj = _gate_proj()
        self.i_conv = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=gate_conv_kernel,
                                groups=self.d_inner, padding=gate_conv_kernel - 1, bias=False)
        self.i_bias = nn.Parameter(torch.full((self.d_inner,), float(input_bias_init)))

        if gate_reparam and gate_rank:
            raise ValueError("gate_reparam is incompatible with gate_rank "
                             "(the rescale targets the full-rank projections)")
        self.gate_reparam = bool(gate_reparam)
        if self.gate_reparam:
            with torch.no_grad():
                self.f_proj.weight.mul_(self.d_inner ** 0.5)
                self.i_proj.weight.mul_(self.d_inner ** 0.5)

    def _gate(self, x, proj, conv, bias, T):
        p = proj(x)
        if self.gate_reparam:
            p = p / (self.d_inner ** 0.5)
        return p + conv(x.transpose(1, 2))[..., :T].transpose(1, 2) + bias

    def forward(self, hidden):
        B, T, _ = hidden.shape
        x, z = self.in_proj(hidden).chunk(2, dim=-1)
        x = F.silu(self.conv1d(x.transpose(1, 2))[..., :T].transpose(1, 2))
        Bm, Cm = torch.split(self.x_proj(x), [self.d_state, self.d_state], dim=-1)
        f_logit = self._gate(x, self.f_proj, self.f_conv, self.f_bias, T)
        i_logit = self._gate(x, self.i_proj, self.i_conv, self.i_bias, T)

        # state readout only (D=0 in the scan); the D skip is applied outside
        cx = naju_scan(x, f_logit, i_logit, Bm, Cm, self.D_zero,
                       backend=self.scan_backend)
        cx = cx * self._cx_scale
        y = (cx + self.D * x) * F.silu(z)
        return self.out_proj(y)


class NajuBlock(nn.Module):
    """Pre-norm residual block: x + Drop(Mixer(RMSNorm(x))). Causal."""

    def __init__(self, d_model, dropout=0.1, use_ckpt=False, **mixer_kw):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = NajuMixer(d_model, **mixer_kw)
        self.drop = nn.Dropout(dropout)
        self.use_ckpt = use_ckpt

    def forward(self, x):
        if self.use_ckpt and torch.is_grad_enabled():
            residual = checkpoint(lambda h: self.drop(self.mixer(self.norm(h))),
                                  x, use_reentrant=False)
        else:
            residual = self.drop(self.mixer(self.norm(x)))
        return x + residual
