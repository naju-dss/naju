"""Shared SSM classifier backbone (used by the Mamba baseline)."""
import torch
import torch.nn as nn

from models.ssm_core import MambaMixer
from data import vocab as V


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class MambaBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1, **mixer_kw):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = MambaMixer(d_model, **mixer_kw)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.mixer(self.norm(x)))


class SSMClassifier(nn.Module):
    """Embedding -> N MambaBlocks -> norm -> last-token MLP head."""

    def __init__(self, vocab_size, num_classes, d_model=256, n_layers=4,
                 dropout=0.1, d_state=16, expand=2, **_):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=V.PAD)
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, dropout=dropout, d_state=d_state, expand=expand)
             for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, num_classes),
        )

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        pooled = x[:, -1]  # query is always the last token
        return self.head(pooled)


def build(cfg, vocab_size, num_classes):
    return SSMClassifier(
        vocab_size=vocab_size, num_classes=num_classes,
        d_model=cfg.get("d_model", 256), n_layers=cfg.get("n_layers", 4),
        dropout=cfg.get("dropout", 0.1), d_state=cfg.get("d_state", 16),
        expand=cfg.get("expand", 2),
    )
