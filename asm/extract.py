"""Assembly101 mistake CSVs -> tidy per-step sequence parquet.

One row per assembly action step, ordered within a sequence, with:
  verb / this / that  = the action = a (verb, two-object) attach/detach move
  label               = correct / mistake / correction (as annotated)
  mistake             = 1 if label == 'mistake'
  mtype               = the remark string for a mistake ('wrong order', ...)
  toy                 = toy id parsed from the filename (assembly grammar differs per toy)
  split               = train/val/test, assigned deterministically per sequence

We keep the *actual* executed order (mistakes and corrections included) so the
history a step is scored against is the real context, exactly as at monitoring time.
"""
from __future__ import annotations

import csv
import hashlib

import pandas as pd

from . import config as C


def _toy_of(stem: str) -> str:
    # nusar-2021_action_both_<actor>-<toy>_<actor>_user_id_<date>_<time>
    try:
        field = stem.split("_")[3]          # e.g. '9022-a18'
        return field.split("-", 1)[1]
    except Exception:
        return "unk"


def _split_of(seq_id: str) -> str:
    h = int(hashlib.md5(seq_id.encode()).hexdigest(), 16) % 100
    if h < 70:
        return "train"
    if h < 85:
        return "val"
    return "test"


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _mtype(remark: str) -> str:
    """Collapse the free-text remark (with typos/variants) into a canonical type."""
    r = _norm(remark)
    if not r:
        return ""
    if "previous" in r:
        return "previous one is mistake"
    if "order" in r or "worng" in r:
        return "wrong order"
    if "position" in r:
        return "wrong position"
    if "happen" in r:
        return "shouldn't have happened"
    return r


def main() -> None:
    files = sorted(C.ANNOTS.glob("*.csv"))
    rows = []
    for f in files:
        seq_id = f.stem
        toy = _toy_of(seq_id)
        split = _split_of(seq_id)
        with open(f, newline="") as fh:
            reader = csv.reader(fh)
            order = 0
            for r in reader:
                if len(r) < 6 or not r[0].strip():
                    continue
                start, end, verb, this, that, label = r[0], r[1], r[2], r[3], r[4], r[5]
                remark = r[6] if len(r) > 6 else ""
                label = _norm(label)
                if label not in ("correct", "mistake", "correction"):
                    continue
                try:
                    start_f, end_f = float(start), float(end)
                except ValueError:
                    start_f = end_f = 0.0
                rows.append(dict(
                    seq_id=seq_id, toy=toy, split=split, order=order,
                    start=start_f, end=end_f,
                    verb=_norm(verb), this=_norm(this), that=_norm(that),
                    label=label,
                    mistake=int(label == "mistake"),
                    correction=int(label == "correction"),
                    mtype=_mtype(remark)))
                order += 1

    df = pd.DataFrame(rows)
    df.to_parquet(C.SEQ_PARQUET)
    n = len(df)
    print(f"rows={n}  sequences={df['seq_id'].nunique()}  toys={df['toy'].nunique()}")
    print("splits:", df["split"].value_counts().to_dict())
    print(f"correct={int((df.label=='correct').sum())}  "
          f"mistake={int(df.mistake.sum())} ({100*df.mistake.mean():.1f}%)  "
          f"correction={int(df.correction.sum())}")
    print("mistake types:", df[df.mistake == 1]["mtype"].value_counts().to_dict())
    print("wrote", C.SEQ_PARQUET)


if __name__ == "__main__":
    main()
