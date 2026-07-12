# Assembly101-NF — 離散の手順ミス検知と反事実推奨

Assembly101 の mistake-detection 注釈（328系列・CC BY-NC 4.0・動画不要）で、各組立手を
`(動詞, 対象, 相手)` の離散トークンとして扱い、履歴で条件づけた自己回帰カテゴリカル
`p(次の手 | これまで, 玩具)` を **correct な手のみ**で学習。SURPRISE=−log p でミスを検知し、
argmax で「本来の次の手」を反事実として推奨する。

**離散×連続の対**の離散側。連続側は
[CaptainCook4D-NF](https://kankusakabe.github.io/CaptainCook4D-NF/)（実行のペースを連続密度で）。

## 結果（差として報告）
- ミス検知 AUROC: AR **0.804** / bigram **0.837** / 連続NF **0.634**
  （離散モデルが連続NFより **+0.171**）。
- 反事実 top-5 的中 **0.340**（候補198手・ランダム≈0.025）。
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
