# Local verification

2026-10-29にWindows CPU環境で実施。JParaCrawl実文はテストに使用していません。

| 確認 | 結果 |
| --- | --- |
| Python 3.11環境 | uv.lockから独立した環境を作成し、pytest 99件成功 |
| Python 3.12環境 | pip editable install、pytest 99件成功 |
| Python 3.13のホスト | WindowsのPython 3.13.14からbootstrapを実行し、ロック済みCPU依存をPython 3.11.18の独立環境へ導入。3.13へ本体依存を導入せず実行できることを確認 |
| Notebook環境分離 | stage CLI・config処理・GPU確認を処理用Pythonへ渡すこと、カーネルから本体やYAML等をimportしないこと、pip fallbackも対応版の処理環境へ限定することをテスト |
| Colab mode safety | smoke / productionの実パス照合、configの保持、モード変更後の古い変数の拒否、各長時間stage直前のpreflightを検証 |
| Colab sampling | head抽出、固定seedのreservoir sampling再現性、既存手動sampleの保持、sampling設定変更時の明示的な再作成要求を確認 |
| Colab realtime logs | mockと実subprocessでstdout / stderrの逐次表示、非ゼロreturn codeの例外、scoring進捗行の表示を確認 |
| GPU bootstrap / preflight | nvidia-smiをmockし、GPU有無によるextras選択と、cuda指定で4項目のいずれかが欠けた場合の停止を検証 |
| Synthetic smoke isolation | mode-specific configでnormalize → rule-filter → speech-filter → exact-dedupまで実行。production入力・config・出力treeの内容が一切変わらないことを確認 |
| Python 3.13の運用ヘルパー | Python 3.13.18でcolab_opsをimportし、本体・PyYAML・fastText・TensorFlow・PyArrowをカーネルへimportしないことを確認 |
| 合成データ全工程 | normalize → rule → speech → exact pair dedup → score double → one-side dedup → select → audit → statsが完了 |
| Resume | 完了generationと推論途中のshardを再利用、出力同一性を確認 |
| 設定変更 / force | target・candidate変更で推論再利用、前stageのforceで子孫を更新 |
| 破損checkpoint | SHA256不一致で停止、forceで別generationを生成 |
| CPU言語判定 | 公式lid.176.ftzとfasttext-wheelを使用し、16件の自作文対に実行。rule通過8件、speech通過7件、exact pair dedup通過6件。再実行時のskipも確認 |
| 300k selection | 310,000件をgeneratorで供給し、300,000件を選択。SQLiteを使用、全件list化なし |
| Bicleaner CLI仕様 | 公式PyPIの3.6 wheelで必要な引数とentry pointを確認 |
| GPU / OOM / CPU fallback | subprocess test doubleで引数・環境変数・batch半減・autoのCPU再試行・cudaの停止を確認 |
| 推論前の完全重複除去 | 完全一致だけを除外し、片側一致の別訳は両方scoreへ渡す。上位candidate枠の浪費解消と、高scoreの別訳が後段で残ることを確認 |
| 完全重複除去の途中再開 | SQLite既出集合を入力から再構築し、保存済みshardを再利用。shard境界を越える重複の再検出を確認 |
| CPU import | TensorFlow・PyTorch・fastTextをimportせずCLIをimport可能 |
| Notebook | nbformat schema、各セルのPython構文、保存outputなしを確認 |
| Figures | 5種類のPNGを生成。score・理由・selected比較図のラベルと余白を目視確認 |
| Git ignore | data / outputs / audit / checkpoint / model / log / TSV / Parquet等が追跡候補から除外されることを確認 |
| Package build | sdist / wheelの作成に成功。sdist内に実データ・モデル・cache・検証出力がないことを確認 |
| 配布入力形式 | NTT公式v3アーカイブをstreamで開き、先頭recordが5列であること、列3が英文・列4が和文（0-based）であることを確認。原文の保存・表示はなし |

この結果はBicleanerの品質評価やL4性能の実測ではありません。合成demoのスコアはtest doubleで、学習データ選定の根拠には使えません。

実BicleanerモデルのCPU / GPU推論、NVIDIA L4の速度・VRAM使用量、ColabのDrive mountと切断後の再同期、JParaCrawl全量からの30万件抽出はローカルでは未検証です。Colab Notebookで確認してください。Notebookのclone先URLは `Yaaamashiro/jparacrawl-filtering` に設定済みです。

## User-reported Colab smoke result

以下は利用者から報告された既存の実データsmoke結果です。今回ローカルで再計測した数値ではありません。

| Stage | Input / accepted | Rejected |
| --- | ---: | ---: |
| Input / normalize | 50,000 | 0 |
| Rule-filter | 38,247 | 11,753 |
| Speech-filter | 36,536 | 1,711 |
| Exact normalized pair dedup | 21,923 | 14,613 |
| Bicleaner processed | 21,923 | — |
| Top candidates | 10,000 | 11,923 |

NVIDIA L4（23034 MiB）、Bicleaner-AI 3.6、GPU実行で約133 pairs/sec、return code 0。大量の完全重複除去はこの入力の観測結果であり、運用改修のために除去規則を変更していません。`src/`、既存bootstrap、default config、uv.lockは今回の変更対象外です。Colab/Drive上の実データ・checkpointへエージェントからの変更は行っていません。

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
