"""Train the AR-categorical head and the dequantised-NF head on 'correct' steps.

Both see the same context encoder and the same training targets (the intended,
successful moves) so the comparison isolates the density family: categorical vs
flow-on-dequantised-tokens.
"""
from __future__ import annotations

import json

import torch

from . import config as C
from . import data as D
from .model import StepModel


def _run(head, tr, va, sizes, epochs=40, bs=256, lr=2e-3):
    torch.manual_seed(C.SEED)
    m = StepModel(sizes, head=head)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    hist, best, best_state = [], float("inf"), None
    for ep in range(epochs):
        m.train()
        tot = n = 0
        for b in D.batches(tr, bs):
            opt.zero_grad()
            loss = m.nll(b, train=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(b["cur_v"]); n += len(b["cur_v"])
        m.eval()
        with torch.no_grad():
            vnll = m.nll(va).item()
        hist.append({"epoch": ep, "train_nll": tot / n, "val_nll": vnll})
        if vnll < best:
            best = vnll
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    return m, {"history": hist, "best_val_nll": best}


def main() -> None:
    sizes = json.loads(C.VOCAB_JSON.read_text())["sizes"]
    tr = D.build(split="train", correct_only=True)
    va = D.build(split="val", correct_only=True)
    print(f"train(correct)={len(tr['cur_v'])}  val(correct)={len(va['cur_v'])}")

    ar, ar_log = _run("ar", tr, va, sizes)
    torch.save(ar.state_dict(), C.AR_PT)
    print(f"AR   best val NLL {ar_log['best_val_nll']:.4f}")

    nf, nf_log = _run("nf", tr, va, sizes)
    torch.save(nf.state_dict(), C.NF_PT)
    print(f"NF   best val NLL {nf_log['best_val_nll']:.4f}")

    C.METRICS_JSON.write_text(json.dumps(
        {"train": {"ar": ar_log, "nf": nf_log,
                   "n_train": len(tr["cur_v"]), "n_val": len(va["cur_v"])}}, indent=1))
    print("wrote", C.METRICS_JSON)


if __name__ == "__main__":
    main()
