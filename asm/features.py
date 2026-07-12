"""Build integer vocabularies for verb / this / that / toy.

Index 0 = padding, index 1 = unknown (unseen at train time). Real symbols start at 2.
Vocab is fit on the TRAIN split only so val/test truly probe generalisation.
"""
from __future__ import annotations

import json

import pandas as pd

from . import config as C

PAD, UNK = 0, 1


def _vocab(series) -> dict:
    vals = sorted(series.dropna().unique().tolist())
    return {v: i + 2 for i, v in enumerate(vals)}


def main() -> None:
    df = pd.read_parquet(C.SEQ_PARQUET)
    tr = df[df["split"] == "train"]
    vocab = {
        "verb": _vocab(tr["verb"]),
        "this": _vocab(tr["this"]),
        "that": _vocab(tr["that"]),
        "toy": _vocab(tr["toy"]),
    }
    sizes = {k: len(v) + 2 for k, v in vocab.items()}
    C.VOCAB_JSON.write_text(json.dumps({"map": vocab, "sizes": sizes}, indent=1))
    print("vocab sizes (incl pad+unk):", sizes)
    print("wrote", C.VOCAB_JSON)


if __name__ == "__main__":
    main()
