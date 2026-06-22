"""Policy networks.

Both take (current word, target word) as letter indices and output a logit
per (position, letter) action, i.e. "change position i to letter c".

PolicyMLP: fixed word length, dumb and fast. The baseline.
PolicyTransformer: per-position tokens, so one model handles any word length.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyMLP(nn.Module):
    def __init__(self, L, hidden=512):
        super().__init__()
        self.L = L
        self.net = nn.Sequential(
            nn.Linear(2 * L * 26, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, L * 26),
        )

    def forward(self, cur, tgt):
        B, L = cur.shape
        x = torch.cat([
            F.one_hot(cur, 26).float().view(B, -1),
            F.one_hot(tgt, 26).float().view(B, -1),
        ], dim=1)
        return self.net(x).view(B, L, 26)


class PolicyTransformer(nn.Module):
    def __init__(self, d=128, heads=4, layers=3, max_len=8):
        super().__init__()
        self.cur_emb = nn.Embedding(26, d)
        self.tgt_emb = nn.Embedding(26, d)
        self.pos_emb = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=4 * d,
            dropout=0.1, batch_first=True, norm_first=True)
        # enable_nested_tensor=False: avoids a noisy warning with norm_first,
        # and our sequences are all the same length anyway (no padding to skip)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, 26)

    def forward(self, cur, tgt):
        # token i = current letter + target letter + position, all at slot i
        pos = torch.arange(cur.shape[1], device=cur.device)
        x = self.cur_emb(cur) + self.tgt_emb(tgt) + self.pos_emb(pos)
        return self.head(self.encoder(x))


def make_model(cfg):
    if cfg["arch"] == "mlp":
        return PolicyMLP(cfg["L"], cfg.get("hidden", 512))
    return PolicyTransformer(cfg.get("d", 128), cfg.get("heads", 4),
                             cfg.get("layers", 3))


def n_params(model):
    return sum(p.numel() for p in model.parameters())
