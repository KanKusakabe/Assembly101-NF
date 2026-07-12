"""Paths and constants for Assembly101-NF (discrete side of the discrete-vs-continuous pair)."""
from __future__ import annotations

from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RAW = DATA / "raw"
ANNOTS = RAW / "md" / "annots"          # cloned assembly101-mistake-detection CSVs
PROC = DATA / "processed"
RESULTS = BASE / "results"
FIGS = RESULTS / "figures"
for _d in (RAW, PROC, RESULTS, FIGS):
    _d.mkdir(parents=True, exist_ok=True)

# CSV columns (no header): start,end,verb,this,that,label,remark
COLS = ["start", "end", "verb", "this", "that", "label", "remark"]

# mistake sub-types live in the `remark` string
MTYPES = ["wrong order", "previous one is mistake",
          "shouldn't have happened", "wrong position"]

HISTORY = 8                # steps of context fed to the GRU
SEED = 20260712

SEQ_PARQUET = PROC / "steps.parquet"
VOCAB_JSON = PROC / "vocab.json"
AR_PT = RESULTS / "ar.pt"
NF_PT = RESULTS / "nf.pt"
METRICS_JSON = RESULTS / "metrics.json"
