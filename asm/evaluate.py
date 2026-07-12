"""What can discrete next-action surprise detect, and how much does the density
family matter here?  Compares AR-categorical, dequantised-NF, and a count-based
bigram baseline on:

  * mistake detection      AUROC(surprise; mistake vs correct)
  * per mistake-type        (wrong order / wrong position / shouldn't-have / prev-mistake)
  * order-violation         synthetic adjacent-swap injection on correct steps
  * anticipation            surprise now vs a mistake within the next 3 steps
  * counterfactual top-k    does the recommended next correct action match the
                            intended (next 'correct') action after a mistake

All numbers are reported as differences ("how much better/worse"), not verdicts.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from . import config as C
from . import data as D
from .model import StepModel

RNG = np.random.default_rng(C.SEED)


def _load(head, sizes):
    m = StepModel(sizes, head=head)
    m.load_state_dict(torch.load(C.AR_PT if head == "ar" else C.NF_PT))
    m.eval()
    return m


def _candidates(tr):
    """Unique (verb,this,that) triples seen among correct train steps."""
    trip = np.stack([tr["cur_v"].numpy(), tr["cur_t"].numpy(), tr["cur_h"].numpy()], 1)
    uniq = np.unique(trip, axis=0)
    return torch.tensor(uniq)


@torch.no_grad()
def _ctx(m, d):
    return m.encoder(d)


@torch.no_grad()
def _ar_surprise(m, d):
    c = _ctx(m, d)
    lv, lt, lh = m.head.logits(c)
    lp = (F.log_softmax(lv, -1).gather(1, d["cur_v"][:, None]).squeeze(1)
          + F.log_softmax(lt, -1).gather(1, d["cur_t"][:, None]).squeeze(1)
          + F.log_softmax(lh, -1).gather(1, d["cur_h"][:, None]).squeeze(1))
    return (-lp).numpy()


@torch.no_grad()
def _nf_surprise(m, d):
    c = _ctx(m, d)
    return (-m.head.log_prob(c, d, train=False)).numpy()


def _bigram(tr):
    """Count model p(action | prev action) with unigram backoff, on correct steps."""
    def key(v, t, h):
        return (int(v), int(t), int(h))
    from collections import defaultdict, Counter
    trans = defaultdict(Counter)
    uni = Counter()
    v, t, h = tr["cur_v"].numpy(), tr["cur_t"].numpy(), tr["cur_h"].numpy()
    hv = tr["hist_v"].numpy()[:, -1]
    ht = tr["hist_t"].numpy()[:, -1]
    hh = tr["hist_h"].numpy()[:, -1]
    for i in range(len(v)):
        a = key(v[i], t[i], h[i])
        p = key(hv[i], ht[i], hh[i])
        trans[p][a] += 1
        uni[a] += 1
    total = sum(uni.values())

    def surprise(d):
        out = []
        V, T, H = d["cur_v"].numpy(), d["cur_t"].numpy(), d["cur_h"].numpy()
        PV, PT, PH = d["hist_v"].numpy()[:, -1], d["hist_t"].numpy()[:, -1], d["hist_h"].numpy()[:, -1]
        for i in range(len(V)):
            a = key(V[i], T[i], H[i]); p = key(PV[i], PT[i], PH[i])
            c = trans.get(p)
            if c and c.get(a):
                pr = c[a] / sum(c.values())
            else:
                pr = (uni.get(a, 0) + 1) / (total + len(uni) + 1)  # add-1 backoff
            out.append(-np.log(pr))
        return np.array(out)
    return surprise


def _auc(score, label):
    if label.sum() == 0 or label.sum() == len(label):
        return float("nan")
    return float(roc_auc_score(label, score))


def _order_injection(ev, m_ar, bigram):
    """On held-out correct steps, swap the last history action for a random other
    train action and measure whether surprise flags the corrupted context."""
    keep = (ev["mistake"] == 0) & (ev["correction"] == 0)
    idx = torch.where(keep)[0]
    base = {k: (v[idx] if torch.is_tensor(v) else v[idx.numpy()]) for k, v in ev.items()}
    n = len(base["cur_v"])
    # corrupted copy: replace the most recent history step with a random train action
    corr = {k: (v.clone() if torch.is_tensor(v) else v.copy()) for k, v in base.items()}
    perm = torch.tensor(RNG.permutation(n))
    corr["hist_v"][:, -1] = base["cur_v"][perm]
    corr["hist_t"][:, -1] = base["cur_t"][perm]
    corr["hist_h"][:, -1] = base["cur_h"][perm]
    s0 = _ar_surprise(m_ar, base)
    s1 = _ar_surprise(m_ar, corr)
    lab = np.concatenate([np.zeros(n), np.ones(n)])
    return _auc(np.concatenate([s0, s1]), lab)


def main() -> None:
    sizes = json.loads(C.VOCAB_JSON.read_text())["sizes"]
    m_ar = _load("ar", sizes)
    m_nf = _load("nf", sizes)

    tr = D.build(split="train", correct_only=True)
    bigram = _bigram(tr)

    # held-out steps = val + test, all labels kept
    va = D.build(split="val")
    te = D.build(split="test")
    ev = {}
    for k in va:
        if torch.is_tensor(va[k]):
            ev[k] = torch.cat([va[k], te[k]])
        else:
            ev[k] = np.concatenate([va[k], te[k]])

    mistake = ev["mistake"].numpy()
    correct = ((ev["mistake"] == 0) & (ev["correction"] == 0)).numpy()
    keep = mistake.astype(bool) | correct           # drop 'correction' rows for detection
    lab = mistake[keep]

    s_ar = _ar_surprise(m_ar, ev)
    s_nf = _nf_surprise(m_nf, ev)
    s_bg = bigram(ev)

    out = {"held_out_n": int(keep.sum()),
           "mistake_rate": float(lab.mean()),
           "detect_mistake": {
               "ar": _auc(s_ar[keep], lab),
               "nf": _auc(s_nf[keep], lab),
               "bigram": _auc(s_bg[keep], lab)}}

    # per mistake-type (mistake rows of that type vs all correct rows)
    mty = ev["mtype"].numpy()
    per = {}
    for i, name in enumerate(C.MTYPES):
        sel = correct | (mistake.astype(bool) & (mty == i))
        per[name] = {
            "n": int((mistake.astype(bool) & (mty == i)).sum()),
            "ar": _auc(s_ar[sel], mistake[sel]),
            "nf": _auc(s_nf[sel], mistake[sel]),
            "bigram": _auc(s_bg[sel], mistake[sel])}
    out["detect_by_type"] = per

    # order-violation injection (AR)
    out["order_injection_auc_ar"] = _order_injection(ev, m_ar, bigram)

    # anticipation: does surprise at a correct step foresee a mistake within 3 steps?
    ant = _anticipation(ev, s_ar, s_nf, s_bg)
    out["anticipation"] = ant

    # counterfactual recommendation
    out["counterfactual"] = _counterfactual(m_ar, tr, ev, bigram)

    metrics = json.loads(C.METRICS_JSON.read_text())
    metrics["evaluate"] = out
    C.METRICS_JSON.write_text(json.dumps(metrics, indent=1))
    print(json.dumps(out, indent=1))
    print("wrote", C.METRICS_JSON)


def _anticipation(ev, s_ar, s_nf, s_bg):
    """Label a *correct* step positive if a mistake occurs within the next 3 steps
    of the same sequence."""
    seq = ev["seq_id"]; order = ev["order"].numpy(); mistake = ev["mistake"].numpy()
    # build per-seq mistake order set
    from collections import defaultdict
    mset = defaultdict(set)
    for i in range(len(seq)):
        if mistake[i]:
            mset[seq[i]].add(int(order[i]))
    correct = (ev["mistake"] == 0) & (ev["correction"] == 0)
    lab, sel = [], []
    for i in range(len(seq)):
        if not bool(correct[i]):
            continue
        o = int(order[i])
        soon = any((o + d) in mset[seq[i]] for d in (1, 2, 3))
        lab.append(int(soon)); sel.append(i)
    sel = np.array(sel); lab = np.array(lab)
    return {"n_pos": int(lab.sum()), "n": int(len(lab)),
            "ar": _auc(s_ar[sel], lab), "nf": _auc(s_nf[sel], lab),
            "bigram": _auc(s_bg[sel], lab)}


@torch.no_grad()
def _counterfactual(m_ar, tr, ev, bigram):
    """For each mistake step, rank candidate actions by success-likelihood and
    check whether the *intended* action (the next 'correct' step in that sequence)
    appears in the top-k.  Baseline = bigram ranking."""
    cand = _candidates(tr)                      # [K,3]
    cand_v, cand_t, cand_h = cand[:, 0], cand[:, 1], cand[:, 2]
    # popularity of each candidate among correct train steps (frequency baseline)
    from collections import Counter as _Ct
    pop = _Ct()
    tv, tt, th = tr["cur_v"].numpy(), tr["cur_t"].numpy(), tr["cur_h"].numpy()
    for i in range(len(tv)):
        pop[(int(tv[i]), int(tt[i]), int(th[i]))] += 1
    cand_pop = np.array([pop[(int(cand_v[k]), int(cand_t[k]), int(cand_h[k]))]
                         for k in range(cand.shape[0])])
    freq_top5 = [tuple(int(x) for x in cand[k]) for k in np.argsort(-cand_pop)[:5]]

    # intended action for each mistake row = the (v,t,h) of the next correct step
    seq = ev["seq_id"]; order = ev["order"].numpy()
    v, t, h = ev["cur_v"].numpy(), ev["cur_t"].numpy(), ev["cur_h"].numpy()
    mistake = ev["mistake"].numpy()
    correct = ((ev["mistake"] == 0) & (ev["correction"] == 0)).numpy()
    from collections import defaultdict
    correct_by_seq = defaultdict(list)
    for i in range(len(seq)):
        if correct[i]:
            correct_by_seq[seq[i]].append((int(order[i]), i))
    for k in correct_by_seq:
        correct_by_seq[k].sort()

    rows = np.where(mistake == 1)[0]
    if len(rows) == 0:
        return {"n": 0}
    d = {k: (val[rows] if torch.is_tensor(val) else val[rows]) for k, val in ev.items()}
    c = m_ar.encoder(d)
    lv, lt, lh = m_ar.head.logits(c)
    lv, lt, lh = F.log_softmax(lv, -1), F.log_softmax(lt, -1), F.log_softmax(lh, -1)
    # joint logprob per candidate: [R,K]
    scoreK = (lv[:, cand_v] + lt[:, cand_t] + lh[:, cand_h])
    topk = torch.topk(scoreK, k=min(5, cand.shape[0]), dim=1).indices.numpy()

    hit1 = hit3 = hit5 = tot = 0
    fq1 = fq3 = fq5 = 0
    for r_i, gi in enumerate(rows):
        s = seq[gi]; o = int(order[gi])
        nxt = [i for (oo, i) in correct_by_seq[s] if oo > o]
        if not nxt:
            continue
        j = nxt[0]
        target = (int(v[j]), int(t[j]), int(h[j]))
        tot += 1
        ranked = topk[r_i]
        got = [(int(cand_v[x]), int(cand_t[x]), int(cand_h[x])) for x in ranked]
        hit1 += int(target == got[0])
        hit3 += int(target in got[:3])
        hit5 += int(target in got[:5])
        fq1 += int(target == freq_top5[0])
        fq3 += int(target in freq_top5[:3])
        fq5 += int(target in freq_top5[:5])
    return {"n": int(tot), "n_candidates": int(cand.shape[0]),
            "random_top5": 5.0 / cand.shape[0],
            "top1_ar": hit1 / tot if tot else float("nan"),
            "top3_ar": hit3 / tot if tot else float("nan"),
            "top5_ar": hit5 / tot if tot else float("nan"),
            "top1_freq": fq1 / tot if tot else float("nan"),
            "top3_freq": fq3 / tot if tot else float("nan"),
            "top5_freq": fq5 / tot if tot else float("nan")}


if __name__ == "__main__":
    main()
