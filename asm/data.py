"""Turn the step parquet into (history -> next-action) training/eval tensors.

For every step t in a sequence we build:
  hist_*  : the previous HISTORY steps' (verb,this,that) ids (left-padded)
  cur_*   : step t's own (verb,this,that) ids  -> the AR target / NF point
  toy     : toy id (conditioning)
  mistake : 1 if step t is an annotated mistake
  mtype   : mistake sub-type index (-1 if not a mistake)
  split   : train/val/test

The density models are trained ONLY on steps whose label == 'correct'
(that is the "intended / successful" distribution). Evaluation scores every
step, so a mistake shows up as low likelihood under the success model.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from . import config as C
from .features import PAD, UNK

MTYPE2IDX = {m: i for i, m in enumerate(C.MTYPES)}


def _load_vocab():
    return json.loads(C.VOCAB_JSON.read_text())


def _id(mapping, key):
    return mapping.get(key, UNK)


def build(split=None, correct_only=False):
    df = pd.read_parquet(C.SEQ_PARQUET)
    vocab = _load_vocab()["map"]
    H = C.HISTORY

    hv, ht, hh = [], [], []          # history verb/this/that
    cv, ct, ch = [], [], []          # current verb/this/that
    toy, mis, cor, mty, spl, sid, order = [], [], [], [], [], [], []

    for seq_id, g in df.groupby("seq_id", sort=False):
        g = g.sort_values("order")
        vs = [_id(vocab["verb"], x) for x in g["verb"]]
        ts = [_id(vocab["this"], x) for x in g["this"]]
        hs = [_id(vocab["that"], x) for x in g["that"]]
        toy_id = _id(vocab["toy"], g["toy"].iloc[0])
        labels = g["label"].tolist()
        mtypes = g["mtype"].tolist()
        sp = g["split"].iloc[0]
        for i in range(len(g)):
            lo = max(0, i - H)
            pv = vs[lo:i]; pt = ts[lo:i]; ph = hs[lo:i]
            pad = H - len(pv)
            hv.append([PAD] * pad + pv)
            ht.append([PAD] * pad + pt)
            hh.append([PAD] * pad + ph)
            cv.append(vs[i]); ct.append(ts[i]); ch.append(hs[i])
            toy.append(toy_id)
            m = int(labels[i] == "mistake")
            mis.append(m)
            cor.append(int(labels[i] == "correction"))
            mty.append(MTYPE2IDX.get(mtypes[i], -1) if m else -1)
            spl.append(sp); sid.append(seq_id); order.append(i)

    d = dict(
        hist_v=torch.tensor(hv), hist_t=torch.tensor(ht), hist_h=torch.tensor(hh),
        cur_v=torch.tensor(cv), cur_t=torch.tensor(ct), cur_h=torch.tensor(ch),
        toy=torch.tensor(toy), mistake=torch.tensor(mis), correction=torch.tensor(cor),
        mtype=torch.tensor(mty), split=np.array(spl), seq_id=np.array(sid),
        order=torch.tensor(order),
    )
    if split is not None:
        mask = d["split"] == split
        d = _subset(d, mask)
    if correct_only:
        # keep only steps annotated 'correct' (the intended, successful build)
        keep = ((d["mistake"] == 0) & (d["correction"] == 0)).numpy()
        d = _subset(d, keep)
    return d


def _subset(d, mask):
    mask_t = torch.tensor(mask) if not torch.is_tensor(mask) else mask
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
        elif torch.is_tensor(v):
            out[k] = v[mask_t]
        else:
            out[k] = v[mask if isinstance(mask, np.ndarray) else mask_t.numpy()]
    return out


def batches(d, bs, shuffle=True):
    n = len(d["cur_v"])
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for s in range(0, n, bs):
        j = idx[s:s + bs]
        yield {k: (v[j] if torch.is_tensor(v) else v[j.numpy()])
               for k, v in d.items() if v is not None}
