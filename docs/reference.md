# 運用リファレンス

## 入力と設定

JParaCrawl v3の日英TSVはヘッダなしの5列です。0始まりの列順は英語ドメイン、日本語ドメイン、配布時のBicleanerスコア、英文、和文です。既定の列指定は `{ja: 4, en: 3}` で、配布時スコアとは別にBicleaner-AIで採点します。展開後のTSVまたは単体gzip圧縮したTSVを指定します。配布TSVは `quoting: false` で読み、CSV形式の引用付きTSVには `quoting: true` を指定します。ヘッダ付き入力は `header: true` と列名で設定します。

| 設定 | production | smoke |
| --- | --- | --- |
| 入力 | `/content/drive/MyDrive/jparacrawl/v3/en-ja/en-ja.bicleaner05.txt` | `/content/drive/MyDrive/jparacrawl/smoke/en-ja-smoke.txt` |
| 設定ファイル | `/content/drive/MyDrive/jparacrawl-filtering/config.yaml` | `/content/jparacrawl-filtering/configs/smoke.yaml` |
| 作業先 | `/content/drive/MyDrive/jparacrawl-filtering` | `/content/drive/MyDrive/jparacrawl-filtering-smoke` |
| 一時作業先 | `/content/jparacrawl-filtering-scratch` | `/content/jparacrawl-filtering-smoke-scratch` |
| 新規設定の目標 / 候補 / 推論シャード | 300,000 / 1,000,000 / 100,000 | 5,000 / 10,000 / 5,000 |

Colabは設定済みのパスとモードの組み合わせを検証し、既存YAMLを再利用します。新規の本番設定は `device: cuda` で生成します。smokeの `head` は先頭行を抽出し、`random` は全入力を走査して固定seedで抽出します。意図して既存のsmoke入力を作り直すときは `REBUILD_SMOKE_INPUT = True` を選びます。

ローカルCLIの相対パスは実行ディレクトリを基準とします。入力の識別にはパス・サイズ・更新時刻を使います。厳密に検証する場合は `input.sha256` を指定すると、起動時に入力全体をハッシュ検証します。完全な設定例は [configs/default.yaml](../configs/default.yaml) にあります。

## フィルタリングと文長の割り当て

1. NFKC、空白と改行の整理、不可視文字の除去を行い、原文も保持します。
2. fastTextの言語判定、文長、文字数比、数字の多重集合、URLやHTML、ナビゲーションや定型文を検査します。
3. 日英の終止符と閉じ括弧・引用符、箇条書き、記号比などから、読み上げに適した文を選びます。
4. 正規化後の両側が一致する文対は入力順で最初の行を採用します。Bicleaner-AIは残りを採点し、スコア順に候補を選びます。
5. 候補からスコア順に採用し、日英いずれかが既採用の文対を除外します。
6. 日本語の文長区分ごとにスコア順で選びます。初期比率は0〜9文字が2%、10〜29文字が15%、30〜59文字が40%、60〜99文字が33%、100文字以上が10%です。丸めや候補不足による空き枠は有効区分全体からスコア順に補充し、目標に届かない場合は不足件数を記録します。

## Bicleaner-AI

モデルの既定値は `bitextor/bicleaner-ai-full-en-ja` です。英文、和文の順でBicleaner CLIに渡し、`--score_only --disable_hardrules` で採点します。モデルとCLIの詳細は配布元を参照してください。[モデル](https://huggingface.co/bitextor/bicleaner-ai-full-en-ja) · [CLI](https://github.com/bitextor/bicleaner-ai)

| `device` | 実行時の動作 |
| --- | --- |
| `cuda` | GPUを必須とし、推論失敗時に停止 |
| `auto` | GPUを試し、未検出または失敗時に警告してCPUへ切り替え |
| `cpu` | CPUで採点 |

GPUメモリ不足時はバッチサイズを半減して再試行します。推論はシャード単位でモデルを読み直し、進捗と残り時間の目安を表示します。Colabでは手順3がGPUの有無から `lid,cuda` または `lid,scoring` を選択し、手順6でworkerとBicleaner CLIの状態を確認します。カーネルとは別にPython 3.11のworker環境を用意します。

再現実験ではモデルのリビジョンを指定してスナップショットを取得し、保存先を `bicleaner.model` に設定して `offline: true` を選びます。

```bash
uv run python scripts/prepare_bicleaner_model.py \
  --revision '<commit-sha>' \
  --destination /content/drive/MyDrive/jparacrawl-filtering/models/bicleaner
```

## CLIとチェックポイント

```bash
uv run python -m jparacrawl_filtering --help
uv run python -m jparacrawl_filtering score --config config.yaml --resume
uv run python -m jparacrawl_filtering select --config config.yaml --resume
uv run python -m jparacrawl_filtering audit --config config.yaml --resume
```

個別の工程を実行すると、前段の整合性を確認して必要な工程から処理します。`run --until exact-dedup` はスコアリング直前で停止します。本番完了後の統計には `scripts/generate_lightweight_stats.py` を使います。パイプラインCLIの `stats` と最後まで実行する `run` では全体分布の計算のため大規模な再走査が発生します。

チェックポイントは `data/<stage>/<generation>/` に保存します。`CURRENT.json` は完了世代、`ACTIVE.json` は処理途中を含む実行対象を示します。`--resume` は完了済みシャードを検証して再利用します。入力・関係する設定・実装・モデル情報・上流世代が変わると、その工程から別世代になります。候補件数・最終件数の変更時は推論結果を再利用できます。`--force speech-filter --resume` は指定工程から後続までの再計算に使います。

中断後はローカルのSQLite索引や完全重複の既出集合を再構築します。gzip入力では先頭から読み直して完了区間を通過します。中間Parquetと世代を保持するため、Driveには入力サイズの数倍を目安に空きを確保してください。同じ作業先への書き込みはランタイム1台で行い、終了前にDriveの同期を待ちます。

## 選抜後の確認と出力

- **監査** — 固定seedで採用500件・不採用200件を抽出します。`outputs/audit/` のCSVで日英の意味対応、Web断片、読み上げ適性を確認します。
- **軽量統計** — 工程の件数と除外理由はmanifestから取得し、候補・重複除去後・最終選抜だけを列指定で読みます。全推論・前半工程の分布は対象外です。`score` と `dedup` では採用行を含むpartを読み、その総行数が工程ごとに110万件を超える場合は読み込み前に停止します。実読込件数は `stats.json` の `io`、文長比較は `length_bins.csv`、要約は `report.md` で確認できます。1文対に複数の除外理由が付くため、理由別件数の合計は除外文対数と異なる場合があります。`--resume` で同じ入力と出力チェックサムに対応する完成済みレポートを再利用します。
- **テキストコーパス** — `training_text` が `normalized` なら `ja_norm` / `en_norm`、`raw` なら `ja_raw` / `en_raw` を使います。設定と選抜時のメタデータを照合し、IDの一意性、空文、train/dev間の同一文対の混入を検証します。dev件数は実際の区分別件数から最大剰余法で配分し、各区分ではseed・bin ID・元IDのSHA256順で選びます。既定は30万件用です。小規模検証では `--expected-total 20 --dev-size 4` のように指定します。`summary.json` はJSONLの保存完了後に書き込まれます。

選抜Parquetは元の行番号、原文、正規化済みテキスト、学習用の `ja` / `en`、フィルタ判定用の特徴量とスコアを含みます。詳しい入力列はJSONL出力時の `summary.json` に記録されます。
