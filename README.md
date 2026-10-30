# jparacrawl-filtering

JParaCrawl v3の日英対訳から、Direct Speech-to-Speech Translation研究のTTS入力候補を抽出するソフトウェアです。対訳品質と発話らしさを別々に評価し、**30–99文字の日本語文を主力に300,000文対**を選びます。閾値と目標件数はYAMLで変更できます。

## Purpose / pipeline

```text
JParaCrawl v3 (TSV / TSV.gz, column mapping)
  → Stage 0 normalize + input validation             CPU
  → Stage 1 language / noise / number / length rules CPU
  → Stage 2 sentence-like / speech suitability       CPU
  → Stage 2b exact normalized pair dedup             CPU
  → Stage 3 Bicleaner-AI → top 1,000,000              GPU or CPU
  → Stage 4 quality-first one-side dedup             CPU
  → Stage 5 length-bin quotas + deficit refill      CPU
  → Stage 6 random manual audit (500 / 200)          CPU
  → Stage 7 statistics + five figures               CPU
```

句読点は文らしさの高Precisionなheuristicです。構文の完全性や対訳の正しさを保証しません。Bicleanerで上位の文でもWeb断片は事前に落とします。TTS・ASR・LLM判定・音声処理は実装範囲外です。

## Requirements

- 主対象: Google Colab、Python 3.11、Google Drive、NVIDIA L4。
- Python 3.11 / 3.12。CPU工程はWindows / Linuxで動作。BicleanerはLinux / Colabで導入してください。
- 基本依存: PyArrow、PyYAML、matplotlib。言語判定: fastText lid.176。Bicleaner: TensorFlow系の独立した子プロセス。
- 全件をPython listに読み込みません。Arrowは通常10,000行、推論は100,000行のshard単位。SQLiteは約64MiBのページキャッシュとディスク索引を使います。推論時のテキストとモデルのメモリはこれに加算されます。
- 全中間結果を保存するため、Driveに入力サイズの数倍以上の空きが必要です。実際の使用量はraw保持・通過率・Parquet圧縮率に依存します。古いgenerationは自動削除しません。

## Installation

リポジトリをcloneし、そのディレクトリで実行します。

```bash
git clone https://github.com/Yaaamashiro/jparacrawl-filtering.git
cd jparacrawl-filtering
```

```bash
uv sync --extra lid --extra dev
uv run python -m jparacrawl_filtering --help
uv run pytest -q
```

`uv.lock` を同梱しています。固定した依存を使う場合は `uv sync --locked ...` を使用してください。

Bicleanerを使うLinux / Colabでは:

```bash
uv sync --extra lid --extra scoring --extra dev
# CUDAライブラリも環境内へ導入する場合は scoring の代わりに cuda:
# uv sync --extra lid --extra cuda --extra dev
```

pip fallback:

```bash
python -m pip install -e '.[lid,scoring,dev]'  # Linux / Colab
# CPUルールだけ: python -m pip install -e '.[lid,dev]'
```

通常のCPU工程・CLI helpではTensorFlow / PyTorchをimportしません。fastTextモデルがない場合に推測ラベルで処理を続けることもありません。NumPy 1.xのfastText wheelとの互換性を保つため、lid extraはNumPyを2未満に制約しています。

## Google Colab usage

[`notebooks/jparacrawl_filtering_colab.ipynb`](notebooks/jparacrawl_filtering_colab.ipynb) を[Colabで開きます](https://colab.research.google.com/github/Yaaamashiro/jparacrawl-filtering/blob/main/notebooks/jparacrawl_filtering_colab.ipynb)。clone先URLは設定済みです。冒頭のセル0で `RUN_MODE = 'smoke'` または `'production'` を選びます。デフォルトはsmokeで、それ以外の文字列は即座にエラーになります。

| 設定 | production | smoke |
| --- | --- | --- |
| config | `/content/drive/MyDrive/jparacrawl-filtering/config.yaml` | `/content/jparacrawl-filtering/configs/smoke.yaml` |
| input | `/content/drive/MyDrive/jparacrawl/v3/en-ja/en-ja.bicleaner05.txt` | `/content/drive/MyDrive/jparacrawl/smoke/en-ja-smoke.txt` |
| work_dir | `/content/drive/MyDrive/jparacrawl-filtering` | `/content/drive/MyDrive/jparacrawl-filtering-smoke` |
| scratch_dir | `/content/jparacrawl-filtering-scratch` | `/content/jparacrawl-filtering-smoke-scratch` |
| target_size（新規設定） | 300,000 | 5,000 |
| candidate_size（新規設定） | 1,000,000 | 10,000 |
| shard_size（新規設定） | 100,000 | 5,000 |

**既存のconfigは上書きしません。** 新規production configは `device: cuda`、`batch_size: auto`、モデル `bitextor/bicleaner-ai-full-en-ja`、`benchmark_pairs: 100000` で作成します。既存の本番configがautoの場合も保存内容は保持し、cudaの推奨を表示します。研究再現runでは、本番configを利用者が明示的に `device: cuda` にしてください。

configのinput / work_dir / scratch_dirとconfig自身のパスは、モードごとの上記パスと照合します。symlinkを解決した実パスで検査し、smokeから本番inputへの取り違え、本番からsmoke work_dirへの取り違え、出力ディレクトリの重なりを拒否します。モードを変更したらセル4のconfig選択もやり直してください。各stage起動時にはconfigを読み直すため、古いNotebook変数の内容では実行しません。

### CPU filtering（GPU不要）

1. セル0で `RUN_MODE = 'production'` を選択。
2. セル1でDrive mount → セル2でclone/pull → セル3でworker bootstrap。
3. セル4でproduction config選択、セル4bで入力・言語モデルの準備。
4. セル5のCPU filteringを実行。直前のpreflightで **production / config / input / work_dir / scratch_dir / target / candidates / device / worker Python / Git commit** を確認します。処理はexact-dedupまでで、configがcudaでもGPUを必要としません。

### GPU scoring（L4へ切替後に再開可能）

1. L4 runtimeへ切り替え、セル0で再度 `RUN_MODE = 'production'` を設定。
2. Drive mount → repo clone/pull → worker bootstrap → production config選択を再実行。
3. bootstrapはnvidia-smiの結果を表示し、GPUありなら **`lid,cuda`**、GPUなしなら **`lid,scoring`** を自動選択します。`USE_CUDA_DEPS` の手動変更は不要です。
4. セル6でGPU preflightを実行。GPU名・VRAM、workerの `cuda_available()`、Bicleaner CLIの絶対パス、package versionを表示します。cuda指定でGPU未検出・workerからのGPU利用不可・CLI/package未導入ならscoringを開始しません。
5. セル7で `score --resume`。起動直前にもpipeline preflightとGPU検査を実施します。既存のCPU stageは現在のsignature検証後に再利用されます。
6. セル8でfinal selection（直前にpreflight表示）→ セル9でaudit → セル10でstats / plot確認。

`run_stage()` はworker Pythonを使い、commandを表示してから `Popen` のstdout / stderrを一行ずつNotebookへflushします。`processed`, `total`, `pairs/sec`, `estimated_remaining_seconds` など既存のscoringログを表示し、非ゼロreturn codeでは例外になります。`shell=True` は使いません。Notebook wrapperで `--config` / `--device` / `--target-size` を追加指定するとpreflightと実行設定がずれるため拒否します。これらの変更は選択中のYAMLに保存してください。通常のCLIのオプションは従来どおりです。

セル10で `module://matplotlib_inline.backend_inline` のbackendエラーになった場合は、`WORKER_ENV["MPLBACKEND"] = "Agg"` を実行してからセル10だけ再実行してください。最新版ではworker起動前にAggを指定します。カーネルのinline表示設定は変更せず、生成したPNGをNotebookに表示します。フィルタリング・scoringの再計算は不要です。

### Smoke test（本番と別work_dir）

セル0を `RUN_MODE = 'smoke'` にし、同じ順で実行します。セル4bは本番TSVから専用smoke入力を作成できます。

- `SMOKE_SAMPLE_MODE = 'head'`（初期値）: 先頭50,000行。高速なpipeline動作確認向けで、サイト分布を代表するsampleではありません。
- `SMOKE_SAMPLE_MODE = 'random'`: seed=42で全入力から50,000行をreservoir sampling。全入力を走査しますがメモリはsample件数に比例し、全件list化しません。採用した行は元の入力順で保存します。
- `SMOKE_SAMPLE_SIZE` / `SMOKE_SEED` で件数・seedを変更できます。原文をNotebookログへ表示しません。
- 既存のsmoke入力は保持します。手動作成された入力はsampling履歴不明と表示して再利用します。自動作成sampleは設定とsourceの識別情報をsidecarへ記録し、同じ設定なら更新しません。違うsampling設定に変更したときは明示的な `REBUILD_SMOKE_INPUT = True` が必要です。入力を作り直すと現在のcheckpoint signature規則に従って再計算されます。
- smoke configはGit除外対象です。GPUランタイムの切替で `/content` のconfigが消えた場合、同じ既定設定なら再作成できます。smokeの閾値等を独自調整した場合はその設定も保存して復元してください。

切断後はモードを再設定し、同じDrive・configでセルを再実行します。各stageには `--resume` を付けています。Drive上にParquetとmanifestが残り、ColabローカルのSQLite scratchは再構築できます。今回の変更ではフィルタ本体、checkpointのcode_digest / config / upstream identityの検証、cudaのfail-fast / auto fallback、OOM時のbatch縮小を変更していません。

ColabのカーネルがPython 3.13でも、セル3が `scripts/bootstrap_colab.py` で処理用のPython 3.11環境を `.work/colab-venv` に作成します。以後の処理とBicleaner CLIはその環境で実行し、Drive mount・表示は元のカーネルで行います。uvによるPython取得・lock済み依存導入を基本とし、pip fallbackも対応バージョンの処理用環境へ限定します。カーネルのPythonやNumPyを上書きしません。`REPO_REF` 未指定ならセル2で既存checkoutを `git pull --ff-only` により更新します。旧Notebookで導入に失敗した場合は最新版を開き直し、セル2以降を再実行してください。

Python 3.13へ処理まで統一しない理由は、[Bicleaner-AI 3.6の配布依存](https://pypi.org/pypi/bicleaner-ai/3.6/json)が `numpy<2` を要求し、[NumPy 1.26の公式対応範囲](https://numpy.org/doc/stable/release/1.26.4-notes.html)がPython 3.9–3.12のためです。Python上限を外すだけでは解決しません。

## JParaCrawl preparation

[NTT公式JParaCrawlページ](https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/)から **English-Japanese training set v3.0** を利用者自身で取得し、Driveへ展開してください。入力にtar.gz自体を指定しないでください。展開した `en-ja/en-ja.bicleaner05.txt`、またはそのファイルを単体gzip化したものを読みます。

[公式v3アーカイブ](https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/bitext/en-ja.tar.gz)の先頭recordで、以下の5列構成を確認しています。実データの文はリポジトリへ保存していません。

v3配布のTSVにはヘッダがなく、次の5列があります。コードの列位置はconfigで変更できます。

| 0-based | 内容 |
| --- | --- |
| 0 | English source domain |
| 1 | Japanese source domain |
| 2 | 配布時のBicleaner score |
| 3 | English text |
| 4 | Japanese text |

配布時スコアは今回のBicleaner-AI scoreとして流用しません。原文に引用符を含むため、デフォルトTSVは `quoting: false` として文字通り読みます。独自にCSV quoting付きTSVを作った場合だけ `quoting: true` にします。列数違いはrecord番号だけを示して停止し、原文をログに出しません。ヘッダあり入力では `header: true` とし、`columns` を列名に変更します。

```yaml
paths:
  input: /content/drive/MyDrive/jparacrawl/v3/en-ja/en-ja.bicleaner05.txt
  work_dir: /content/drive/MyDrive/jparacrawl-filtering
  scratch_dir: /content/jparacrawl-filtering-scratch
input:
  columns: {ja: 4, en: 3}  # zero-based
```

相対パスはCLIを実行したディレクトリ基準です。Driveでは絶対パスを推奨します。入力のpath / size / mtimeをfingerprintに含めます。より厳密な入力識別には `input.sha256` に既知のSHA256を設定してください。設定時は起動ごとに全入力を検証するため追加I/Oが必要です。

fastTextのモデル取得:

```bash
uv run python -m jparacrawl_filtering download-lid --config configs/default.yaml
```

公式の [lid.176.ftz](https://fasttext.cc/docs/en/language-identification.html) を `language_id.model_path` へ保存します。ソフトウェアMITとは別のモデル利用条件があります。

## Configuration

完全な設定例は [`configs/default.yaml`](configs/default.yaml)。`validate-config` で比率・範囲・列mappingを検査できます。

| 設定 | デフォルト / 方針 |
| --- | --- |
| `target_size` | 300,000 |
| `seed` | 42 |
| `batch_size` | CPU 10,000 rows |
| `language_id.min_confidence` | 0.8、ja / enの両方で検査 |
| `length` | ja 4–300文字、en 2–150語、緩い文字数比0.05–5.0 |
| `numbers.enabled` | true |
| `numbers.require_both_sides` | true、両側に数字がある場合だけmultiset不一致をreject |
| `symbols.max_weird_symbol_ratio` | 0.10 |
| `sentence.require_terminal_punctuation` | true、閉じ引用符・括弧を許可 |
| `bicleaner.candidate_size` | 1,000,000、thresholdではなく降順ranking |
| `bicleaner.batch_size` | auto、GPU32 / CPU8から開始しOOMで半減 |
| `bicleaner.shard_size` | 100,000、切断時の推論再計算単位 |
| `bicleaner.device` | autoは警告付きCPU fallback、cudaはGPU必須・失敗時停止 |
| `one_side_dedup` | true、exact pair dedupは常時有効 |
| `training_text` | normalized、rawも選べる |
| `audit` | accepted 500 / rejected 200 |

長さbin初期値: 0–9=2%、10–29=15%、30–59=40%、60–99=33%、100以上=10%。極端な短文は4文字未満・2英単語未満を落とし、残る短文は少量のquotaで扱います。binは0から連続し最後は `max: null`。ratio合計は0より大きく1以下です。ratio=0のbinは補充対象からも除きます。

binごとの `floor(target × ratio)` をscore降順に採用し、不足分・丸めの余り・ratio合計が1未満の残枠は有効bin全体のscore順で補充します。候補が足りない場合は件数を水増しせず警告し、reportにshortfallを記録します。候補が十分なら指定件数を満たします。枠再配分後の比率は初期quotaから変わる場合があります。

## CLI usage

```bash
uv run python -m jparacrawl_filtering validate-config --config configs/default.yaml
uv run python -m jparacrawl_filtering run --config configs/default.yaml --resume
uv run python -m jparacrawl_filtering run --config configs/default.yaml --resume --until speech-filter
uv run python -m jparacrawl_filtering score --config configs/default.yaml --resume --device cpu
uv run python -m jparacrawl_filtering select --config configs/default.yaml --resume --target-size 300000
uv run python -m jparacrawl_filtering audit --config configs/default.yaml --resume
uv run python -m jparacrawl_filtering stats --config configs/default.yaml --resume
uv run python -m jparacrawl_filtering run --config configs/default.yaml --resume --force speech-filter
```

stage名: `normalize`, `rule-filter`, `speech-filter`, `exact-dedup`, `score`, `dedup`, `select`, `audit`, `stats`。stage単体コマンドも必要な前段の整合性を確認し、未実行・変更された前段を処理します。`--resume` で有効な前段はskipします。statsはauditも生成します。`--force` はstage名を取り、複数回指定できます。

## Resume behavior

各stageは `data/<stage>/<generation>/manifest.json` と複数のParquetを持ちます。小さいmanifestとParquetは一時ファイルへ書いた後でrenameします。完了したshardのSHA256とArrow footer件数を確認して再利用します。完了済みgenerationは `CURRENT.json`、途中も含む実行対象は `ACTIVE.json` が示します。

- 通常の再実行には `--resume` を付けます。同一設定でcheckpointがあるのに付けなければ停止します。
- 入力・関連設定・コードfingerprint・モデル情報・上流generationが変わると、そのstageから別generationを作成します。target_size変更で推論をやり直しません。candidate_size変更でも推論shardは再利用します。
- `--force speech-filter --resume` はspeech-filterを再計算し、その子孫も再計算します。normalizeとrulesは再利用します。
- CPU変換・推論はshard単位に再開。gzip入力は先頭から読み直して完了済み区間を飛ばすため、読み取り時間はかかります。
- 全体rankingのSQLite索引とquota計算は切断後に再構築します。保存済みの推論は繰り返しません。
- `exact-dedup` は途中再開時に入力を先頭からstreamし、SQLiteの既出集合を再構築します。保存済みParquet shardは検証して再利用します。
- 破損checkpointは黙ってskipせず停止します。該当stageをforceしてください。
- work_dir内のOS lockで同時書き込みを拒否します。lock非対応のマウントでは停止します。Driveを複数Colabから同時操作しないでください。
- Driveクライアントへの書込完了とクラウド同期完了は別です。ランタイムを意図的に終了する前にDrive同期を待ってください。

## Output format

```text
work_dir/
  data/
    normalize/<generation>/part-*.parquet
    rule-filter/<generation>/part-*.parquet
    speech-filter/<generation>/part-*.parquet
    exact-dedup/<generation>/part-*.parquet # 正規化後の完全重複を推論前に除外
    scoring/<generation>/part-*.parquet     # 推論全結果の再開用
    score/<generation>/part-*.parquet       # top candidates + quality_rank rejects
    dedup/<generation>/part-*.parquet
    select/<generation>/part-*.parquet      # bin選択・不採用の両方
  checkpoints/pipeline.lock
  outputs/
    selected/selected.parquet              # accepted=true の学習用のみ
    audit/accepted_sample.csv
    audit/rejected_sample.csv
    stats/stats.json
    stats/stages.csv
    stats/reject_reasons.csv
    stats/effective_config.json
    stats/histograms.json
    stats/report.md
    stats/*.png
```

最終の `selected.parquet` は `ja` / `en`、rawとnormalized両方、言語判定確率、文字数・英単語数・length_ratio、各種ノイズfeature、文末判定、数字整合性、Bicleaner score、length_bin、pair_id、accepted、reject_reason、first_reject_stage、stageを保持します。`estimated_duration` / `actual_duration` はnullable floatとして予約し、将来の音声秒数選択に拡張できます。

各stageの出力はそのstageに入った行とreject理由を保存し、次stageはacceptedだけを読みます。全行の最終状態を知りたい場合は各stageのrejectedと最終selectedを合わせます。前stageのrejectを次stageへ複製しません。pair_idは入力record順で安定、score同点はpair_id昇順です。

## Filtering rules

- NFKC、両端空白除去、連続空白・改行の統合、制御・不可視format文字除去。rawを併記します。
- fastTextでja / enを確認し、confidence不足も理由を残します。
- URL / email / HTML entity・tag / 代表的なファイル名・絶対パスをreject。
- breadcrumbはseparator数と密度の両方を要求。単独の `>` だけでrejectしません。
- boilerplateは短文または©とkeywordの組合せ。長文の規約への言及をkeywordだけで落としません。
- コード・数式・Unicode symbol比を計測。通常の句読点や引用符は異常記号から除きます。
- 箇条書きはrawの改行を保持した段階でfeature化。`- Item` は箇条書き、ハイフン直後に空白がない語と区別します。先頭 `- ` の引用会話も落ち得るPrecision重視のルールです。
- 日英の終止符と閉じ括弧・引用符を確認。
- 数字はArabic numeralをdecimalに揃えて多重集合で比較。片側の数字欠落はdefaultで不問です。漢数字・数詞・日付書式・単位変換の完全対応はしません。
- `exact-dedup` は正規化後の `(ja_norm, en_norm)` が完全一致する文対だけを推論前に除外し、入力順で最初の文対を残します。rawだけが異なる場合も、同じ推論用テキストなら重複扱いです。片側だけ一致する別訳は残します。
- Bicleaner後の `dedup` はscore順で貪欲に採用。日本語または英語が既採用なら除外します。最大二部matchingではありません。完全重複を前段で除くため、同じ文対がquality candidate枠を浪費しません。

`reject_reason` はsemicolon区切りで複数保存します。`lang_mismatch`, `lang_low_confidence`, `too_short`, `too_long`, `url`, `email`, `html`, `file_path`, `navigation`, `boilerplate`, `code_like`, `formula_like`, `symbol_ratio`, `bullet_like`, `numeric_mismatch`, `not_sentence_like`, `duplicate_pair`, `duplicate_ja`, `duplicate_en` に加え、`empty_text`, `length_ratio`, `quality_rank`, `length_quota` を使用します。

## Bicleaner scoring

モデルは [`bitextor/bicleaner-ai-full-en-ja`](https://huggingface.co/bitextor/bicleaner-ai-full-en-ja)。[公式CLI](https://github.com/bitextor/bicleaner-ai)へ **English→Japanese** の順で渡し、`--score_only --disable_hardrules` で確率を受け取ります。独自ルールとBicleaner hardrulesの混同を避け、KenLM / Hunspellの追加導入を不要にしています。

子プロセスでのみTensorFlowを使用します。GPUはnvidia-smiで事前確認し、GPU実行時には公式CLIの `--require_gpu` でTensorFlowからの利用可否も確認します。`device: cuda` はGPUが未検出ならRuntimeErrorで停止し、GPU実行が失敗した場合もCPUへ切り替えません。`device: auto` はGPU未検出・GPU実行失敗を警告し、CPUで再試行します。`device: cpu` は初めからCPUで実行します。

OOMはGPU上でbatch sizeを半分にして再試行します。1でも失敗した場合、cudaは停止、autoは警告してCPUへ移ります。auto batch sizeは安全な初期値からの縮小のみで、VRAMを埋めるための増加探索はしません。100k行ごとにthroughput・残り件数に基づくETAを記録します。モデル読込時間もthroughputに含みます。L4の使用を必須にする実験では `--device cuda` を指定してください。

100k shardごとにモデルを子プロセスで読み直すためロード費用が発生します。`shard_size` を増やすと効率が改善する場合がありますが、切断時の再計算範囲とテキスト保持量も増えます。自動的なCPU fallbackは遅くなるため、ログのdeviceを確認してください。

再現実験ではremote modelのmainを直接使い続けず、commit SHAのsnapshotをDriveへ固定することを推奨します:

```bash
uv run python scripts/prepare_bicleaner_model.py \
  --destination /content/drive/MyDrive/jparacrawl-filtering/models/bicleaner
# 表示されたディレクトリを bicleaner.model に設定し offline: true にする。
# 以降 --revision <記録したSHA> で同一モデルを取得可能。
```

モデルディレクトリのファイル名・サイズ・mtimeとBicleanerのパッケージ版をcheckpoint識別に含めます。remote IDの場合はモデル更新の自動検知ができません。scoreは閾値で最終選択せず、candidate_size内の順位とbin比率で選択します。

## Manual audit

seed固定のreservoir samplingにより、採用500件・全stageの不採用200件を均等ランダム抽出します。候補が少なければ全件を出力し、重複sampleは作りません。日英の意味対応・Web断片・読み上げ適性を確認してからTTSへ進んでください。CSVには原文が含まれるため `outputs/` ごとGit対象外です。

## Statistics

stageごとのinput / accepted / rejected / reject_rateと理由別件数をmanifestに保存し、最後にJSON・CSVへまとめます。理由は複数付き得るため、理由別件数の和はreject行数を超えることがあります。単語数は英字語・数値を数える軽量regexです。

matplotlibで日本語文字数・英語語数（before / after rules / selected比較）、Bicleaner分布、selected vs rejected長さ分布、reject理由件数の5図を生成します。ヒストグラムは全行をstreamして固定binへ集計し、図の最後のbinは上限超過を含みます。設定、bin定義、選択件数・shortfall、実行時間・throughputも保存します。原文は統計や通常ログに書きません。

## Tests / offline synthetic demo

```bash
uv run pytest -q
uv run ruff check src scripts tests
uv run python scripts/validate_notebook.py
uv run python scripts/synthetic_demo.py
uv run python scripts/synthetic_demo.py --resume
# 任意: 31万件の合成候補から30万件を選択するCPU検証
uv run python scripts/benchmark_selection.py
```

テストfixtureは完全自作のJSONです。合成demoは **言語判定とscoringにtest doubleを注入** し、ダウンロード・GPUなしで全stage / Parquet / audit / figures / resumeを確認します。demoのscoreはBicleaner評価結果として使用できません。CLI設定に疑似scoring backendはありません。

pytestは正常文・各種reject・引用文・数字・dedup・quotaと補充・CPU import・OOM / CPU fallback・途中切断再開・force伝播・破損checkpoint・lockを検証します。NotebookはschemaとセルのPython構文を検査します。実L4・Drive mount・実Bicleanerモデルの推論はColabでの検証が必要です。ローカル検証の実施範囲は [`docs/verification.md`](docs/verification.md) に記録します。

## License / data and copyright notice

自作ソフトウェアは [MIT License](LICENSE)。JParaCrawl・fastTextモデル・Bicleanerおよびそのモデルには各々の条件が適用されます。

This repository contains filtering software only and does not redistribute JParaCrawl data.

Users must obtain JParaCrawl v3.0 from the [official source](https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/) and comply with its terms of use.

Derived datasets generated using this repository may also be subject to the JParaCrawl terms of use and the rights associated with the original web content.

JParaCrawlデータがMIT Licenseになるわけではありません。公式利用条件を確認してください。原データ・派生データ・監査実文・checkpoint・モデル・ログを再配布しません。`.gitignore` はこれらのディレクトリとParquet / TSV / CSV / archive / model / logを除外します。Notebookに実文出力を保存しないでください。
