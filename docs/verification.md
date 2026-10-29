# Local verification

2026-10-29にWindows CPU環境で実施。JParaCrawl実文はテストに使用していません。

| 確認 | 結果 |
| --- | --- |
| Python 3.11環境 | uv.lockから独立した環境を作成し、pytest 44件成功 |
| Python 3.12環境 | pip editable install、pytest 44件成功 |
| 合成データ全工程 | normalize → rule → speech → score double → dedup → select → audit → statsが完了 |
| Resume | 完了generationと推論途中のshardを再利用、出力同一性を確認 |
| 設定変更 / force | target・candidate変更で推論再利用、前stageのforceで子孫を更新 |
| 破損checkpoint | SHA256不一致で停止、forceで別generationを生成 |
| CPU言語判定 | 公式lid.176.ftzとfasttext-wheelを使用し、16件の自作文対に実行。rule通過8件、speech通過7件。再実行時のskipも確認 |
| 300k selection | 310,000件をgeneratorで供給し、300,000件を選択。SQLiteを使用、全件list化なし |
| Bicleaner CLI仕様 | 公式PyPIの3.6 wheelで必要な引数とentry pointを確認 |
| GPU / OOM / CPU fallback | subprocess test doubleで引数・環境変数・batch半減・CPU再試行を確認 |
| CPU import | TensorFlow・PyTorch・fastTextをimportせずCLIをimport可能 |
| Notebook | nbformat schema、各セルのPython構文、保存outputなしを確認 |
| Figures | 5種類のPNGを生成。score・理由・selected比較図のラベルと余白を目視確認 |
| Git ignore | data / outputs / audit / checkpoint / model / log / TSV / Parquet等が追跡候補から除外されることを確認 |
| Package build | sdist / wheelの作成に成功。sdist内に実データ・モデル・cache・検証出力がないことを確認 |
| 配布入力形式 | NTT公式v3アーカイブをstreamで開き、先頭recordが5列であること、列3が英文・列4が和文（0-based）であることを確認。原文の保存・表示はなし |

この結果はBicleanerの品質評価やL4性能の実測ではありません。合成demoのスコアはtest doubleで、学習データ選定の根拠には使えません。

実BicleanerモデルのCPU / GPU推論、NVIDIA L4の速度・VRAM使用量、ColabのDrive mountと切断後の再同期、JParaCrawl全量からの30万件抽出はローカルでは未検証です。Colab Notebookで確認してください。Notebookのclone先URLは `Yaaamashiro/jparacrawl-filtering` に設定済みです。

再実行コマンド:

```bash
uv sync --locked --extra lid --extra dev
uv run pytest -q
uv run ruff check src scripts tests
uv run python scripts/validate_notebook.py
uv run python scripts/synthetic_demo.py
uv run python scripts/synthetic_demo.py --resume
uv run python scripts/benchmark_selection.py
```
