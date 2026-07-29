"""Naju v1 classifier for the synthetic suite.

Embedding -> N NajuBlocks (from the naju reference package) -> RMSNorm ->
last-token MLP head, mirroring the SSM/FLA classifier backbones so the
comparison is controlled. Defaults follow the paper configuration
(d_model=128, d_state=64, b_f=+5, b_i=-2, D init 0.01).
"""
import os
import sys

import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))  # repo root (for naju)
from naju import NajuBlock, RMSNorm
from data import vocab as V


class NajuClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=128, n_layers=4,
                 dropout=0.1, **mixer_kw):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=V.PAD)
        self.blocks = nn.ModuleList(
            [NajuBlock(d_model, dropout=dropout, **mixer_kw)
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
        return self.head(self.norm(x)[:, -1])


def build(cfg, vocab_size, num_classes):
    return NajuClassifier(
        vocab_size=vocab_size, num_classes=num_classes,
        d_model=cfg.get("d_model", 128), n_layers=cfg.get("n_layers", 4),
        dropout=cfg.get("dropout", 0.1),
        d_state=cfg.get("d_state", 64), expand=cfg.get("expand", 2),
        gate_conv_kernel=cfg.get("gate_conv_kernel", 4),
        forget_bias_init=cfg.get("forget_bias_init", 5.0),
        input_bias_init=cfg.get("input_bias_init", -2.0),
        d_init=cfg.get("d_init", 0.01),
    )
