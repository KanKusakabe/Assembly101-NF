"""Figures + Japanese index.html / README for Assembly101-NF.

Language policy: describe *how much better or worse* one approach is than another
(differences in AUROC / accuracy), not winners and losers.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
for _f in ("Hiragino Sans", "Hiragino Maru Gothic Pro", "Arial Unicode MS", "YuGothic"):
    try:
        from matplotlib import font_manager as _fm
        if any(ff.name == _f for ff in _fm.fontManager.ttflist):
            plt.rcParams["font.family"] = _f
            break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from . import data as D
from .model import StepModel

AC = "#c2410c"      # accent (this project = discrete)
AC2 = "#2563eb"     # blue (the continuous/NF contrast)
GY = "#9ca3af"


def _m():
    return json.loads(C.METRICS_JSON.read_text())


def fig_detect(m):
    d = m["evaluate"]["detect_mistake"]
    names = ["AR\n(離散カテゴリカル)", "bigram\n(遷移カウント)", "NF\n(連続・脱量子化)"]
    vals = [d["ar"], d["bigram"], d["nf"]]
    cols = [AC, "#f59e0b", AC2]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    bars = ax.bar(names, vals, color=cols, width=.62)
    ax.axhline(.5, ls="--", c=GY, lw=1); ax.text(2.35, .51, "偶然=0.5", color=GY, fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + .01, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1); ax.set_ylabel("ミス検知 AUROC")
    ax.set_title("手順ミスの検知：離散モデルと連続NFの差")
    fig.tight_layout(); fig.savefig(C.FIGS / "detect.png", dpi=130); plt.close(fig)


def fig_bytype(m):
    per = m["evaluate"]["detect_by_type"]
    order = ["shouldn't have happened", "wrong order", "wrong position", "previous one is mistake"]
    jp = {"shouldn't have happened": "不要な取り外し", "wrong order": "順序ミス",
          "wrong position": "位置ミス", "previous one is mistake": "連鎖ミス"}
    labels = [f"{jp[k]}\n(n={per[k]['n']})" for k in order]
    ar = [per[k]["ar"] for k in order]
    bg = [per[k]["bigram"] for k in order]
    nf = [per[k]["nf"] for k in order]
    x = np.arange(len(order)); w = .26
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(x - w, ar, w, label="AR", color=AC)
    ax.bar(x, bg, w, label="bigram", color="#f59e0b")
    ax.bar(x + w, nf, w, label="NF", color=AC2)
    ax.axhline(.5, ls="--", c=GY, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1); ax.set_ylabel("AUROC"); ax.legend(fontsize=9)
    ax.set_title("ミス種別ごとの検知しやすさ")
    fig.tight_layout(); fig.savefig(C.FIGS / "bytype.png", dpi=130); plt.close(fig)


def fig_curves(m):
    ar = m["train"]["ar"]["history"]; nf = m["train"]["nf"]["history"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, h, name, col in ((axes[0], ar, "AR（離散・カテゴリカル尤度）", AC),
                             (axes[1], nf, "NF（連続・脱量子化密度）", AC2)):
        ax.plot([e["epoch"] for e in h], [e["train_nll"] for e in h], c=col, label="train")
        ax.plot([e["epoch"] for e in h], [e["val_nll"] for e in h], c=col, ls="--", label="val")
        ax.set_title(name, fontsize=10); ax.set_xlabel("epoch"); ax.set_ylabel("NLL"); ax.legend(fontsize=8)
    fig.suptitle("学習曲線（AR と NF は基底測度が異なり NLL は直接比較不可）", fontsize=10)
    fig.tight_layout(); fig.savefig(C.FIGS / "curves.png", dpi=130); plt.close(fig)


def fig_counterfactual(m):
    c = m["evaluate"]["counterfactual"]
    ks = ["top1", "top3", "top5"]
    ar = [c[f"{k}_ar"] for k in ks]; fq = [c[f"{k}_freq"] for k in ks]
    rnd = c["random_top5"]
    x = np.arange(3); w = .36
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.bar(x - w / 2, ar, w, label="尤度で推奨 (AR)", color=AC)
    ax.bar(x + w / 2, fq, w, label="頻度で推奨", color=GY)
    ax.axhline(rnd, ls=":", c="#444", lw=1)
    ax.text(2.1, rnd + .01, f"ランダム top5≈{rnd:.3f}", fontsize=8, color="#444")
    ax.set_xticks(x); ax.set_xticklabels(["top-1", "top-3", "top-5"])
    ax.set_ylabel("意図した次手の的中率")
    ax.set_title(f"反事実の手順推奨（ミス直後 n={c['n']}／候補{c['n_candidates']}手）")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig(C.FIGS / "counterfactual.png", dpi=130); plt.close(fig)


@torch.no_grad()
def fig_timeline(m):
    """Per-step surprise timeline for one illustrative held-out sequence."""
    sizes = json.loads(C.VOCAB_JSON.read_text())["sizes"]
    ar = StepModel(sizes, head="ar"); ar.load_state_dict(torch.load(C.AR_PT)); ar.eval()
    ev = D.build(split="test")
    seqs = np.unique(ev["seq_id"])
    # choose the sequence with the most mistakes for a legible example
    best, bestn = None, -1
    for s in seqs:
        idx = np.where(ev["seq_id"] == s)[0]
        nm = int(ev["mistake"].numpy()[idx].sum())
        if nm > bestn:
            best, bestn = s, nm
    idx = np.where(ev["seq_id"] == best)[0]
    d = {k: (v[idx] if torch.is_tensor(v) else v[idx]) for k, v in ev.items()}
    c = ar.encoder(d)
    lv, lt, lh = ar.head.logits(c)
    lp = (F.log_softmax(lv, -1).gather(1, d["cur_v"][:, None]).squeeze(1)
          + F.log_softmax(lt, -1).gather(1, d["cur_t"][:, None]).squeeze(1)
          + F.log_softmax(lh, -1).gather(1, d["cur_h"][:, None]).squeeze(1))
    surprise = (-lp).numpy()
    o = d["order"].numpy(); mis = d["mistake"].numpy(); cor = d["correction"].numpy()
    srt = np.argsort(o)
    surprise, mis, cor = surprise[srt], mis[srt], cor[srt]
    x = np.arange(len(surprise))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(x, surprise, c="#333", lw=1.3, marker="o", ms=3)
    for i in x:
        if mis[i]:
            ax.axvline(i, c=AC, alpha=.35, lw=6)
        elif cor[i]:
            ax.axvline(i, c="#16a34a", alpha=.25, lw=6)
    ax.set_xlabel("組立ステップ順"); ax.set_ylabel("サプライズ = −log p(次手)")
    ax.set_title(f"1系列のサプライズ推移（橙=ミス, 緑=修正）  toy={str(best).split('_')[3]}")
    fig.tight_layout(); fig.savefig(C.FIGS / "timeline.png", dpi=130); plt.close(fig)


def write_html(m):
    e = m["evaluate"]; dm = e["detect_mistake"]; cf = e["counterfactual"]
    d_ar, d_bg, d_nf = dm["ar"], dm["bigram"], dm["nf"]
    gap = d_ar - d_nf
    html = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Assembly101-NF — 離散の手順ミス検知と反事実推奨</title>
<style>
:root{{--acc:#c2410c;--acc2:#2563eb;--ink:#222;--mut:#666;--bg:#faf8f5}}
*{{box-sizing:border-box}}
body{{font:16px/1.75 -apple-system,"Hiragino Sans","Noto Sans JP",sans-serif;color:var(--ink);max-width:960px;margin:0 auto;padding:2rem 1rem;background:#fff}}
a{{color:var(--acc)}}
h1{{line-height:1.3;margin:.2rem 0;font-size:1.9rem}}
.sub{{color:var(--mut);font-size:1.03rem}}
.lead{{background:var(--bg);border-left:4px solid var(--acc);padding:1rem 1.2rem;border-radius:0 10px 10px 0;margin:1.4rem 0}}
h2{{border-left:6px solid var(--acc);padding-left:.6rem;font-size:1.3rem;margin-top:2.2rem}}
img{{max-width:100%;border:1px solid #eee;border-radius:10px;margin:.6rem 0}}
table{{border-collapse:collapse;width:100%;font-size:.93rem;margin:.6rem 0}}
th,td{{border:1px solid #e5e5e5;padding:.4rem .6rem;text-align:center}}
th{{background:#f7f5f1}}
code{{background:#f3f1ec;padding:.1rem .35rem;border-radius:4px;font-size:.9em}}
.k{{color:var(--acc);font-weight:700}}
footer{{color:var(--mut);font-size:.85rem;margin-top:3rem;border-top:1px solid #eee;padding-top:1rem}}
</style></head><body>

<h1>Assembly101-NF</h1>
<p class="sub">離散の「次の手」尤度で手順ミスを検知し、反事実として正しい手を推奨する — <b>離散×連続の対</b>の離散側</p>

<div class="lead">
玩具組立（Assembly101 mistake-detection・328系列）の各手を <code>(動詞, 対象, 相手)</code> の離散トークンとして、
履歴で条件づけた自己回帰カテゴリカル <b>p(次の手 | これまで, 玩具)</b> を <b>correct な手のみ</b>で学習。
<b>サプライズ = −log p</b> でミスを検知し、<b>argmax</b> で「本来の次の手」を反事実として推奨する。
姉妹プロジェクト <a href="https://kankusakabe.github.io/CaptainCook4D-NF/">CaptainCook4D-NF</a>（連続・実行のペース）と対にして、
<b>「尤度による失敗検知でどの密度モデルが向くかは信号の性質で変わる」</b>を検証する。
</div>

<h2>要点</h2>
<ul>
<li><b>離散のミス検知は自己回帰カテゴリカルが素直</b>：ミス検知 AUROC は AR <span class="k">{d_ar:.3f}</span>／
遷移カウント bigram <b>{d_bg:.3f}</b>。同じトークンを連続化して流し込む NF は <span style="color:var(--acc2)">{d_nf:.3f}</span> で、
離散モデルより <b>{gap:+.3f}</b>（NF側が低い）。<b>連続密度のNFを離散手順にそのまま当てるのは不利</b>。</li>
<li><b>信号は主に局所的</b>：単純な bigram が AR とほぼ同等〜わずかに上。手順ミスの多くは直前の手との整合で説明でき、
履歴や玩具の条件づけの上積みは小さい（正直な観察）。</li>
<li><b>NLLは直接比較しない</b>：AR は離散分布の対数尤度、NF は連続密度（スケール依存）で<b>基底測度が異なる</b>。
比較は下流の AUROC で行う。</li>
<li><b>反事実推奨は機能</b>：ミス直後に「本来の次手」を top-5 で <span class="k">{cf['top5_ar']:.3f}</span> 的中
（候補 {cf['n_candidates']} 手・ランダム top5≈{cf['random_top5']:.3f}）。頻度推奨より高い。</li>
</ul>

<h2>手順ミスの検知</h2>
<img src="results/figures/detect.png" alt="detection AUROC">
<p>ミス種別で見ると、<b>不要な取り外し</b>は離散モデルで AUROC ~0.97 と明確に検知でき、
<b>順序ミス・位置ミス</b>は 0.5〜0.68 とはるかに難しい。「間違い」の検知しやすさは種別に強く依存する。</p>
<img src="results/figures/bytype.png" alt="AUROC by mistake type">

<h2>反事実：本来の次の手を推奨</h2>
<p>各ミス直後に、成功尤度 <code>p(次手|履歴)</code> の高い手を順位づけし、その系列で実際に続いた
<b>correct な次手</b>を的中できるかを測る。尤度による推奨は頻度ベースラインより高く、ランダムを大きく上回る。</p>
<img src="results/figures/counterfactual.png" alt="counterfactual top-k">

<h2>1系列のサプライズ推移</h2>
<img src="results/figures/timeline.png" alt="surprise timeline">
<p>橙＝ミス、緑＝修正。ミス（特に不要な取り外し）でサプライズが持ち上がる様子が見える。</p>

<h2>学習曲線</h2>
<img src="results/figures/curves.png" alt="training curves">

<h2>再現</h2>
<pre><code>uv run python -m asm.extract     # CSV -> steps.parquet
uv run python -m asm.features    # 語彙
uv run python -m asm.train       # AR + 脱量子化NF
uv run python -m asm.evaluate    # 検知/種別/反事実
uv run python -m asm.report      # 図 + このindex.html</code></pre>

<footer>
データ: <a href="https://github.com/assembly-101/assembly101-mistake-detection">Assembly101 mistake-detection</a>
（CC BY-NC 4.0・注釈のみ使用）。KAN-NF シリーズ。自動生成。
</footer>
</body></html>"""
    (C.BASE / "index.html").write_text(html)
    print("wrote index.html")


def write_readme(m):
    e = m["evaluate"]; dm = e["detect_mistake"]; cf = e["counterfactual"]
    txt = f"""# Assembly101-NF — 離散の手順ミス検知と反事実推奨

Assembly101 の mistake-detection 注釈（328系列・CC BY-NC 4.0・動画不要）で、各組立手を
`(動詞, 対象, 相手)` の離散トークンとして扱い、履歴で条件づけた自己回帰カテゴリカル
`p(次の手 | これまで, 玩具)` を **correct な手のみ**で学習。SURPRISE=−log p でミスを検知し、
argmax で「本来の次の手」を反事実として推奨する。

**離散×連続の対**の離散側。連続側は
[CaptainCook4D-NF](https://kankusakabe.github.io/CaptainCook4D-NF/)（実行のペースを連続密度で）。

## 結果（差として報告）
- ミス検知 AUROC: AR **{dm['ar']:.3f}** / bigram **{dm['bigram']:.3f}** / 連続NF **{dm['nf']:.3f}**
  （離散モデルが連続NFより **{dm['ar']-dm['nf']:+.3f}**）。
- 反事実 top-5 的中 **{cf['top5_ar']:.3f}**（候補{cf['n_candidates']}手・ランダム≈{cf['random_top5']:.3f}）。
- AR と NF の NLL は基底測度が異なり直接比較しない。比較は下流 AUROC で行う。

## 再現
```
uv run python -m asm.extract
uv run python -m asm.features
uv run python -m asm.train
uv run python -m asm.evaluate
uv run python -m asm.report
```
公開: https://kankusakabe.github.io/Assembly101-NF/
"""
    (C.BASE / "README.md").write_text(txt)
    print("wrote README.md")


def main() -> None:
    m = _m()
    fig_detect(m); fig_bytype(m); fig_curves(m); fig_counterfactual(m); fig_timeline(m)
    write_html(m); write_readme(m)
    print("report done")


if __name__ == "__main__":
    main()
