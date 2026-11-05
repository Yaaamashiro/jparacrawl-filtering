# JParaCrawl Filtering

JParaCrawl v3の日英対訳から、Direct Speech-to-Speech Translation研究向けの学習コーパスを作ります。対訳の品質と発話らしさで絞り込み、日本語30〜99文字の文を中心に **30万文対** を選びます。

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yaaamashiro/jparacrawl-filtering/blob/main/notebooks/jparacrawl_filtering_colab.ipynb)

- **選抜** — 正規化・ルール判定 → 完全重複除去 → Bicleaner-AI → 片側重複除去 → 文長別の割り当て
- **再開** — Driveに保存したチェックポイントを使って処理を継続
- **出力** — 選抜済みParquet、確認用CSV、集計レポート、学習用JSONL

## Colabで使う

1. JParaCrawl v3.0の **English–Japanese training set** を入手します。[NTT公式配布ページ](https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/)
2. 配布アーカイブを展開し、`en-ja.bicleaner05.txt` をDriveの `MyDrive/jparacrawl/v3/en-ja/` に置きます。
3. ノートブックを開き、目的に合う手順を実行します。番号はノートブックの見出しに対応します。

| 目的 | 実行する手順 |
| --- | --- |
| 初回の本番処理 | 0〜4bで準備 → 5でCPUフィルタリング → 6〜7でGPU確認とスコアリング → 8で選抜 |
| 小規模に動作確認 | 0で `RUN_MODE = 'smoke'` を選び、1〜8を実行 |
| 切断・GPUへの切り替え後 | 0〜4で同じ設定を読み込み、続きの手順へ。スコアリングから再開する場合は6〜7 |
| 選抜結果を確認 | 9で監査サンプル、10で軽量集計 |
| **選抜済みParquetをJSONLに変換** | **0〜4 → 11**。CPUランタイムで実行可能 |

本番のスコアリングにはNVIDIA L4を想定しています。ノートブックは処理用のPython 3.11環境を用意します。既定モードは `production` です。`smoke` は別の入力と作業先を使い、既定では先頭5万行から最大5,000文対を選びます。

本番の設定と成果物は `MyDrive/jparacrawl-filtering/` に保存されます。モードを切り替えたら手順4で設定を選び直します。入力・設定・再開条件の詳細は[運用リファレンス](docs/reference.md)を参照してください。

## 成果物

| 作業ディレクトリからのパス | 内容 |
| --- | --- |
| `outputs/selected/selected.parquet` | 最終選抜の文対。原文、正規化済みテキスト、ID、スコアを収録 |
| `outputs/audit/*.csv` | 採用・不採用から抽出した人手確認用の文対 |
| `outputs/stats/report.md` | 通過件数、除外理由、文長・スコアの分布。CSV・JSON・PNGも同じディレクトリに保存 |
| `outputs/text-corpus/manifest.jsonl` | 学習用の文対と `train` / `dev` の割り当て |
| `outputs/text-corpus/summary.json` | 件数、採用したテキスト列、入力・出力のSHA256、文長区分別の分割件数 |

### 学習用JSONL

手順11は選抜済み30万文対を読み、既定のseed `42` で **train 298,000件・dev 2,000件** に分けます。各文長区分の比率に合わせてdevを選び、元の `pair_id` を引き継ぎます。`training_text` に従って正規化済みテキストまたは原文を出力します。同じ入力と設定・seedで再実行すると同じJSONLになります。

```json
{"pair_id":"p000012345678","split":"train","ja_text":"これは例文です。","en_text":"This is an example.","corpus":"jparacrawl_v3_filtered","length_bin":2,"bicleaner_score":0.998,"source_line":12345678}
```

## ローカルで使う

Python 3.11または3.12と `uv` を使います。CPU工程と合成データの動作確認はWindows / Linuxに対応します。Bicleaner-AIを使う場合はLinux / Colabで `--extra scoring` を追加します。

```bash
git clone https://github.com/Yaaamashiro/jparacrawl-filtering.git
cd jparacrawl-filtering
uv sync --locked --extra lid --extra dev
uv run python scripts/synthetic_demo.py
```

実データ処理では [configs/default.yaml](configs/default.yaml) をコピーし、入力・作業先・モデルのパスを環境に合わせます。下の例ではコピーを `config.yaml` としています。

```bash
uv run python -m jparacrawl_filtering validate-config --config config.yaml
uv run python -m jparacrawl_filtering download-lid --config config.yaml
uv run python -m jparacrawl_filtering run --config config.yaml --resume --until exact-dedup
```

選抜済み成果物から統計やJSONLだけを生成する場合は、既存の設定ファイルを指定します。

```bash
uv run python scripts/generate_lightweight_stats.py --config config.yaml --resume
uv run python scripts/export_text_corpus.py --config config.yaml --seed 42
```

設定、CLI、再開の詳細は[運用リファレンス](docs/reference.md)にまとめています。

## 開発

```bash
uv run pytest -q
uv run ruff check src scripts tests
uv run python scripts/build_notebook.py
uv run python scripts/validate_notebook.py
```

Colabノートブックの編集元は [scripts/build_notebook.py](scripts/build_notebook.py) です。ローカルでの確認範囲と実行結果は[検証記録](docs/verification.md)に記載しています。

## ライセンスとデータ

ソフトウェアはMIT Licenseです。[LICENSE](LICENSE)

JParaCrawlの原データと派生成果物を扱う際は、公式配布元の利用条件と元のWebコンテンツの権利を確認してください。原データ、派生データ、監査CSV、チェックポイントは利用者の環境で管理します。モデルにはそれぞれの利用条件が適用されます。[JParaCrawl公式配布ページ](https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/)
