"""Next-action density over a discrete assembly step, conditioned on history.

ConditionEncoder: GRU over the previous steps' (verb,this,that) embeddings +
a toy embedding -> a context vector for predicting the NEXT step's action.

Two interchangeable heads share one interface so we can ask, on genuinely
*discrete* data, how an autoregressive categorical model compares with a
normalizing flow forced onto the same tokens (the discrete-vs-continuous point):

  * ARHead   -- factorised categorical  p(verb)·p(this|ctx)·p(that|ctx).
  * NFHead   -- zuko conditional NSF over the 3 token ids made continuous by
                uniform dequantisation (the flow's natural home is continuous).

SURPRISE = -log p(action | history).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import zuko

DIM = 3  # (verb, this, that)


class ConditionEncoder(nn.Module):
    def __init__(self, sizes, emb=24, gru_hidden=96, out_dim=96, toy_emb=16):
        super().__init__()
        self.ev = nn.Embedding(sizes["verb"], emb, padding_idx=0)
        self.et = nn.Embedding(sizes["this"], emb, padding_idx=0)
        self.eh = nn.Embedding(sizes["that"], emb, padding_idx=0)
        self.etoy = nn.Embedding(sizes["toy"], toy_emb, padding_idx=0)
        self.gru = nn.GRU(3 * emb, gru_hidden, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden + toy_emb, out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim), nn.ReLU())
        self.out_dim = out_dim

    def forward(self, b):
        seq = torch.cat([self.ev(b["hist_v"]), self.et(b["hist_t"]),
                         self.eh(b["hist_h"])], dim=-1)
        _, h = self.gru(seq)
        h = h.squeeze(0)
        return self.mlp(torch.cat([h, self.etoy(b["toy"])], dim=-1))


class ARHead(nn.Module):
    """Factorised categorical head over (verb, this, that)."""

    def __init__(self, ctx_dim, sizes):
        super().__init__()
        self.fv = nn.Linear(ctx_dim, sizes["verb"])
        self.ft = nn.Linear(ctx_dim, sizes["this"])
        self.fh = nn.Linear(ctx_dim, sizes["that"])

    def logits(self, c):
        return self.fv(c), self.ft(c), self.fh(c)

    def log_prob(self, c, b):
        lv, lt, lh = self.logits(c)
        return (F.log_softmax(lv, -1).gather(1, b["cur_v"][:, None]).squeeze(1)
                + F.log_softmax(lt, -1).gather(1, b["cur_t"][:, None]).squeeze(1)
                + F.log_softmax(lh, -1).gather(1, b["cur_h"][:, None]).squeeze(1))


class NFHead(nn.Module):
    """Conditional NSF over uniformly-dequantised token ids (continuous surrogate)."""

    def __init__(self, ctx_dim, transforms=3, hidden=(64, 64)):
        super().__init__()
        self.flow = zuko.flows.NSF(features=DIM, context=ctx_dim,
                                   transforms=transforms, hidden_features=hidden)

    @staticmethod
    def _dequant(b, train):
        x = torch.stack([b["cur_v"], b["cur_t"], b["cur_h"]], -1).float()
        if train:
            x = x + torch.rand_like(x)      # uniform dequantisation
        else:
            x = x + 0.5
        return x / 10.0                     # mild scaling for stable spline range

    def log_prob(self, c, b, train=False):
        x = self._dequant(b, train)
        return self.flow(c).log_prob(x)


class StepModel(nn.Module):
    def __init__(self, sizes, head="ar"):
        super().__init__()
        self.encoder = ConditionEncoder(sizes)
        self.kind = head
        self.head = ARHead(self.encoder.out_dim, sizes) if head == "ar" \
            else NFHead(self.encoder.out_dim)

    def log_prob(self, b, train=False):
        c = self.encoder(b)
        if self.kind == "ar":
            return self.head.log_prob(c, b)
        return self.head.log_prob(c, b, train=train)

    def nll(self, b, train=False):
        return -self.log_prob(b, train=train).mean()
