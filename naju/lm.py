"""Causal language model with Naju v1 mixer blocks.

Embedding + N pre-norm causal blocks + final RMSNorm + tied LM head — the
configuration used for the paper's WikiText-103 experiments.
"""
import torch
import torch.nn as nn

from naju.mixer import NajuBlock, RMSNorm


class NajuLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=6, dropout=0.1,
                 tie_weights=True, use_ckpt=False, **mixer_kw):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            NajuBlock(d_model, dropout=dropout, use_ckpt=use_ckpt, **mixer_kw)
            for _ in range(n_layers)])
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
