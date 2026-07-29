"""Baseline sequence models (HGRN, GLA, RWKV6, RetNet, Mamba-2) wrapped in the
same classifier backbone used for Naju: embedding -> N mixer blocks (RMSNorm +
token-mixer + residual, no separate FFN, matching our SSM blocks) -> last-token
head. Mixers come from flash-linear-attention and mamba-ssm.
"""
import os
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn

from data import vocab as V
from models.backbone import RMSNorm

def _make_mixer(kind, d_model, n_heads, layer_idx, short_conv=True):
    # A short depthwise conv before the token mixer is essential for associative
    # recall (Zoology/Based); mamba and naju both have one, so we enable it for
    # the linear-attention baselines too for a fair comparison. RWKV6 instead uses
    # its built-in token-shift, so it has no use_short_conv knob.
    # fla imported lazily (like mamba_ssm/xlstm below) so Mamba/Naju runs work on
    # machines without flash-linear-attention installed.
    if kind in ("gla", "hgrn", "rwkv", "retnet"):
        import fla.layers as FL
    if kind == "gla":
        return FL.GatedLinearAttention(hidden_size=d_model, num_heads=n_heads,
                                       use_short_conv=short_conv)
    if kind == "hgrn":
        return FL.HGRNAttention(hidden_size=d_model, use_short_conv=short_conv,
                                layer_idx=layer_idx)
    if kind == "rwkv":
        return FL.RWKV6Attention(hidden_size=d_model, num_heads=n_heads, layer_idx=layer_idx)
    if kind == "retnet":
        return FL.MultiScaleRetention(hidden_size=d_model, num_heads=n_heads,
                                      use_short_conv=short_conv)
    if kind == "mamba2":
        # imported lazily so the other baselines work on machines without mamba-ssm
        from mamba_ssm import Mamba2
        return Mamba2(d_model=d_model, d_state=64, headdim=32)
    if kind == "xlstm":
        # Importing the xlstm package pulls in the sLSTM CUDA-init module, which
        # needs CUDA_HOME just to resolve include paths (even though we only use the
        # pure-PyTorch mLSTM and never compile the sLSTM kernel). Point it at a
        # harmless existing dir so the import succeeds on clean envs without a
        # CUDA toolkit.
        os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", "/usr"))
        from xlstm import mLSTMLayer, mLSTMLayerConfig
        return mLSTMLayer(mLSTMLayerConfig(
            embedding_dim=d_model, context_length=4096, num_heads=n_heads))
    raise ValueError(kind)


class _Block(nn.Module):
    def __init__(self, kind, d_model, n_heads, layer_idx, dropout):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = _make_mixer(kind, d_model, n_heads, layer_idx)
        self.kind = kind
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.mixer(self.norm(x))
        if isinstance(y, tuple):
            y = y[0]
        return x + self.drop(y)


class FLAClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, kind, d_model=128, n_layers=4,
                 n_heads=4, dropout=0.1, **_):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=V.PAD)
        self.blocks = nn.ModuleList(
            [_Block(kind, d_model, n_heads, l, dropout) for l in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x)[:, -1])


def build(cfg, vocab_size, num_classes):
    return FLAClassifier(
        vocab_size=vocab_size, num_classes=num_classes,
        kind=cfg["kind"],
        d_model=cfg.get("d_model", 128),
        n_layers=cfg.get("n_layers", 4),
        n_heads=cfg.get("n_heads", 4),
        dropout=cfg.get("dropout", 0.1),
    )
