"""Causal LM sharing the exact Naju mixer used across the suite; Mamba /
Mamba-2 baselines come from the official mamba-ssm package.

A causal block is a pre-norm residual with the mixer applied left-to-right only;
all mixers are causal by construction (left-padded convs sliced to [:T], and a
left-to-right scan), so there is no future leakage. Embedding + N causal blocks
+ final norm + tied LM head.
"""
import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))   # repo root (for naju)
from naju import NajuMixer, RMSNorm


def make_mixer(backbone, d_model, cfg):
    if backbone == "naju":
        return NajuMixer(
            d_model, d_state=cfg.get("d_state", 64), expand=2,
            gate_conv_kernel=4, forget_bias_init=5.0, input_bias_init=-2.0,
            d_init=cfg.get("d_init", 0.01))
    if backbone == "mamba":
        from mamba_ssm import Mamba
        return Mamba(d_model, d_state=cfg.get("d_state", 16),
                     d_conv=4, expand=2)
    if backbone == "mamba2":
        from mamba_ssm import Mamba2
        return Mamba2(d_model, d_state=cfg.get("d_state", 64),
                      d_conv=4, expand=2, headdim=64)
    raise ValueError(backbone)


class CausalBlock(nn.Module):
    """Pre-norm residual, causal mixer."""

    def __init__(self, d_model, mixer, dropout=0.1):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = mixer
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.mixer(self.norm(x)))


def _rope(q, k):
    """Rotary position embedding on the head dim (q,k: [B,H,T,Dh], Dh even)."""
    B, H, T, Dh = q.shape
    inv = 1.0 / (10000.0 ** (torch.arange(0, Dh, 2, device=q.device, dtype=torch.float32) / Dh))
    ang = torch.arange(T, device=q.device, dtype=torch.float32)[:, None] * inv[None, :]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]           # [1,1,T,Dh/2]

    def rot(x):
        x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
        out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return out.flatten(-2).to(x.dtype)
    return rot(q), rot(k)


class TransformerBlock(nn.Module):
    """Standard pre-norm decoder block: causal RoPE self-attention + 4x GELU FFN."""

    def __init__(self, d_model, n_heads=4, dropout=0.1, ffn_mult=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.norm1 = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model))
        self.drop = nn.Dropout(dropout)
        self.attn_dropout = dropout

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = (t.view(B, T, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        q, k = _rope(q, k)
        o = nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0)
        x = x + self.drop(self.proj(o.transpose(1, 2).reshape(B, T, D)))
        return x + self.drop(self.ffn(self.norm2(x)))


class CausalLM(nn.Module):
    def __init__(self, vocab_size, backbone="naju", d_model=256, n_layers=6,
                 dropout=0.1, tie_weights=True, cfg=None):
        super().__init__()
        cfg = cfg or {}
        self.embed = nn.Embedding(vocab_size, d_model)
        if backbone == "transformer":
            self.blocks = nn.ModuleList([
                TransformerBlock(d_model, n_heads=cfg.get("n_heads", 4), dropout=dropout)
                for _ in range(n_layers)])
        else:
            self.blocks = nn.ModuleList([
                CausalBlock(d_model, make_mixer(backbone, d_model, cfg), dropout)
                for _ in range(n_layers)
            ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # GPT-style small init: keeps initial logits ~O(1) so init loss ~ log(vocab)
        # instead of blowing up (critical once the head is tied to the embedding).
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        if tie_weights:
            self.head.weight = self.embed.weight

    def forward(self, ids):
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))          # [B, T, vocab]


def build_lm(cfg):
    return CausalLM(
        vocab_size=cfg["vocab_size"], backbone=cfg.get("backbone", "naju"),
        d_model=cfg.get("d_model", 256), n_layers=cfg.get("n_layers", 6),
        dropout=cfg.get("dropout", 0.1), tie_weights=cfg.get("tie_weights", True),
        cfg=cfg)
