"""Transformer encoder baseline classifier with rotary or learned positions."""
import torch
import torch.nn as nn

from data import vocab as V


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        d_model=256,
        n_layers=4,
        n_heads=4,
        ffn_dim=1024,
        dropout=0.1,
        max_len=4096,
        **_,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=V.PAD)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, num_classes),
        )

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos(pos)
        pad_mask = input_ids == V.PAD  # [B, T] True == ignore
        if not pad_mask.any():
            pad_mask = None            # no padding: avoid the (B*H, L, L) mask expansion
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)
        pooled = x[:, -1]  # query is the last token
        return self.head(pooled)


def build(cfg, vocab_size, num_classes):
    return TransformerClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=cfg.get("d_model", 256),
        n_layers=cfg.get("n_layers", 4),
        n_heads=cfg.get("n_heads", 4),
        ffn_dim=cfg.get("ffn_dim", 1024),
        dropout=cfg.get("dropout", 0.1),
        max_len=cfg.get("max_len", 4096),
    )
