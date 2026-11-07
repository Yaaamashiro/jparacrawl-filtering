"""Regenerate the output-free Colab notebook from reviewable Python cell sources."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""# JParaCrawlの日英対訳を選抜する

JParaCrawl v3から発話に適した文対を選び、DriveにParquetと学習用JSONLを保存します。

| 目的 | 実行する手順 |
| --- | --- |
| 初回の本番処理 | **0〜4b** 準備 → **5** フィルタリング → **6〜7** GPU確認と採点 → **8** 選抜 |
| 小規模に試す | **0** で `smoke` を選択 → **1〜8** |
| 切断・ランタイム切り替え後 | **0〜4** で同じ設定を読み込み、続きの手順へ |
| 選抜結果を確認 | **9** 監査サンプル、**10** 軽量統計 |
| **選抜済みParquetをJSONLへ変換** | **0〜4 → 11**。CPUで実行できます |

初回処理では、展開済みの `en-ja.bicleaner05.txt` をDriveの `MyDrive/jparacrawl/v3/en-ja/` に置きます。本番のBicleaner採点にはL4を想定しています。手順5をCPUで済ませてから、GPUに切り替えられます。

[リポジトリと入力データの案内](https://github.com/Yaaamashiro/jparacrawl-filtering#using-google-colab)
""")
md("""## 0. 実行モードを選ぶ

- `production`（既定）で本番の設定と成果物を使います。
- `smoke` では別の作業先に小規模な入力を作ります。`head` は先頭行、`random` は入力全体から固定seedで抽出します。

モードを切り替えたら手順4で設定を選び直します。
""")
code("""
RUN_MODE = 'production'  # 'production' or 'smoke'; select smoke for a small pipeline test.
SMOKE_SAMPLE_MODE = 'head'  # 'head': fast pipeline check; 'random': scans the full input.
SMOKE_SAMPLE_SIZE = 50000
SMOKE_SEED = 42
REBUILD_SMOKE_INPUT = False  # Preserve existing smoke data unless explicitly requested.
if RUN_MODE not in ('smoke', 'production'):
    raise ValueError("RUN_MODE must be 'smoke' or 'production'")
if SMOKE_SAMPLE_MODE not in ('head', 'random'):
    raise ValueError("SMOKE_SAMPLE_MODE must be 'head' or 'random'")
""")
md("## 1. Google Driveを接続する")
code("""
from google.colab import drive
drive.mount('/content/drive')
""")
md("""## 2. リポジトリを取得・更新する

既存のcheckoutは `git pull --ff-only` で更新します。再現実験では `REPO_REF` にコミットSHAまたはタグを指定できます。
""")
code("""
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = 'https://github.com/Yaaamashiro/jparacrawl-filtering.git'
REPO_REF = ''  # Optional commit SHA or tag for reproducibility.
REPO_DIR = Path('/content/jparacrawl-filtering')
if not REPO_URL:
    raise ValueError('Set REPO_URL to the published repository URL before running.')
if not REPO_DIR.exists():
    subprocess.run(['git', 'clone', REPO_URL, str(REPO_DIR)], check=True)
else:
    origin = subprocess.check_output(['git', '-C', str(REPO_DIR), 'remote', 'get-url', 'origin'], text=True).strip()
    if origin != REPO_URL:
        raise ValueError('Existing checkout has a different remote; choose another REPO_DIR.')
    if not REPO_REF:
        subprocess.run(['git', '-C', str(REPO_DIR), 'pull', '--ff-only'], check=True)
if REPO_REF:
    subprocess.run(['git', '-C', str(REPO_DIR), 'checkout', REPO_REF], check=True)
os.chdir(REPO_DIR)
import importlib
sys.path.insert(0, str(REPO_DIR / 'scripts'))
import colab_ops as ops
importlib.reload(ops)
LAYOUT = ops.path_layout(REPO_DIR)
print('Python:', sys.version.split()[0])
print('Commit:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
""")
md("""## 3. 処理用のPython環境を用意する

ノートブックのカーネルとは別にPython 3.11の環境を作り、検出したGPUに合わせて依存関係を導入します。ランタイム切り替え後もこの手順を実行します。
""")
code("""
gpu = ops.nvidia_gpu()
extras = ops.gpu_extras(gpu)
print('GPU detected:', gpu or 'none')
print('Installing extras:', extras)
ops.stream_command([sys.executable, 'scripts/bootstrap_colab.py', '--extras', extras], env=os.environ.copy())
import json
runtime = json.loads((REPO_DIR / '.work/colab_environment.json').read_text())
PIPELINE_PYTHON = runtime['python']
# Only subprocesses use the worker environment; the notebook kernel stays intact.
WORKER_ENV = os.environ.copy()
for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'):
    WORKER_ENV.pop(key, None)
WORKER_ENV['PATH'] = str(Path(PIPELINE_PYTHON).parent) + os.pathsep + WORKER_ENV.get('PATH', '')
WORKER_ENV['PYTHONUNBUFFERED'] = '1'
WORKER_ENV['PYTHONIOENCODING'] = 'utf-8'
WORKER_ENV['MPLBACKEND'] = 'Agg'  # Save figures without inheriting Colab's inline backend.
print('Notebook kernel:', sys.version.split()[0])
print('Pipeline worker:', runtime['version'], PIPELINE_PYTHON)
""")
md("""## 4. 設定と保存先を確認する

表示されるモード、設定ファイル、入力、作業先を確認します。本番ではDriveの既存YAMLを再利用します。新規の本番設定はGPU必須の `device: cuda` です。
""")
code("""
CONFIG = LAYOUT[ops.validate_mode(RUN_MODE)]['config']
def read_active_config():
    prepared = json.loads(subprocess.check_output([
        PIPELINE_PYTHON, 'scripts/prepare_colab_config.py', '--mode', RUN_MODE,
        '--repository', str(REPO_DIR), '--config', str(CONFIG),
    ], text=True, env=WORKER_ENV))
    return prepared['config']

config = read_active_config()
ops.validate_run(RUN_MODE, CONFIG, config, LAYOUT)
WORK = Path(config['paths']['work_dir'])
INPUT = Path(config['paths']['input'])
print('Selected mode:', RUN_MODE, '\\nConfig:', CONFIG, '\\nInput:', INPUT, '\\nWork:', WORK)
if RUN_MODE == 'production' and config['bicleaner']['device'] != 'cuda':
    print('Research runs recommend device: cuda. Existing config is preserved; edit it explicitly if needed.')

def run_stage(stage, *options):
    current = read_active_config()  # Do not rely on stale notebook variables.
    ops.validate_run(RUN_MODE, CONFIG, current, LAYOUT)
    command = ops.pipeline_command(PIPELINE_PYTHON, stage, CONFIG, options)
    if stage in ('run', 'score', 'select'):
        ops.preflight(RUN_MODE, CONFIG, current, LAYOUT, PIPELINE_PYTHON, WORKER_ENV, REPO_DIR)
    if ops.will_score(stage, options):
        ops.gpu_preflight(PIPELINE_PYTHON, current, WORKER_ENV)
    return ops.stream_command(command, env=WORKER_ENV)
""")
md("""## 4b. 入力と言語判定モデルを準備する

初回フィルタリングの前に実行します。smokeでは入力を抽出し、両モードで入力の存在と設定を確認してfastTextモデルを用意します。
""")
code("""
config = read_active_config()
ops.validate_run(RUN_MODE, CONFIG, config, LAYOUT)
if RUN_MODE == 'smoke':
    ops.create_smoke_sample(LAYOUT['production']['input'], LAYOUT['smoke']['input'], LAYOUT,
                           sample_mode=SMOKE_SAMPLE_MODE, size=SMOKE_SAMPLE_SIZE,
                           seed=SMOKE_SEED, overwrite=REBUILD_SMOKE_INPUT)
else:
    print('Production mode: smoke sampling skipped.')
if not Path(config['paths']['input']).is_file():
    raise FileNotFoundError('Prepare the extracted JParaCrawl TSV on Drive first.')
run_stage('validate-config')
run_stage('download-lid')
""")
md("""## 5. CPUでフィルタリングする

正規化、ルール・発話らしさの判定、完全重複除去まで進めます。完了後にL4へ切り替える場合は、**0〜4 → 6** と進みます。
""")
code("run_stage('run', '--resume', '--until', 'exact-dedup')")
md("""## 6. スコアリング環境を確認する

GPUの検出結果と、処理用Pythonから見えるCUDA・Bicleaner CLIを確認します。`device: cuda` でGPUを使えない場合はここで停止します。
""")
code("""
config = read_active_config()
ops.validate_run(RUN_MODE, CONFIG, config, LAYOUT)
ops.gpu_preflight(PIPELINE_PYTHON, config, WORKER_ENV)
""")
md("""## 7. Bicleaner-AIで採点する

候補を採点し、進捗と残り時間の目安を表示します。初回はモデルを取得します。中断後は手順0〜4と6を実行し、ここから再開します。
""")
code("run_stage('score', '--resume')")
md("""## 8. 文対を選抜する

片側重複を除き、文長区分ごとに選抜します。成果物は作業先の `outputs/selected/selected.parquet` です。
""")
code("""
run_stage('select', '--resume')
# target_size is saved in the Drive config; changing it preserves inference shards.
""")
md("""## 9. 人手確認用のサンプルを作る

既定で採用500件・不採用200件のCSVを `outputs/audit/` に保存します。
""")
code("""
run_stage('audit', '--resume')
print('Review:', WORK / 'outputs/audit')
""")
md("""## 10. 選抜結果を集計する

完了済みのチェックポイントから通過件数と分布を集計します。CPUで実行でき、レポートと図は `outputs/stats/` に保存します。
""")
code("""
config = read_active_config()
ops.validate_run(RUN_MODE, CONFIG, config, LAYOUT)
command = [PIPELINE_PYTHON, str(REPO_DIR / 'scripts/generate_lightweight_stats.py'),
           '--config', str(CONFIG), '--resume']
ops.stream_command(command, env=WORKER_ENV)
from IPython.display import Image, display
STATS = Path(config['paths']['work_dir']) / 'outputs/stats'
summary = json.loads((STATS / 'stats.json').read_text(encoding='utf-8'))
print(f"選抜 {summary['selected_size']:,} / 目標 {summary['target_size']:,} 文対")
print('レポート', STATS / 'report.md')
for name in ['stage_retention.png', 'selected_ja_length.png', 'selected_en_words.png',
             'selected_bicleaner_scores.png', 'length_bin_comparison.png',
             'candidate_bicleaner_scores.png', 'reject_reasons.png']:
    display(Image(filename=str(STATS / name)))
""")
md("""## 11. 学習用JSONLを出力する

本番の選抜済み30万文対を、元の `pair_id` を保ったまま **train 298,000件・dev 2,000件** に分けます。seedは42です。手順0〜4で本番設定を読み込めば、選抜済みParquetから直接実行できます。

`outputs/text-corpus/manifest.jsonl` と `summary.json` をDriveに保存します。
""")
code("""
if RUN_MODE == 'production':
    config = read_active_config()
    ops.validate_run(RUN_MODE, CONFIG, config, LAYOUT)
    ops.stream_command([
        PIPELINE_PYTHON, str(REPO_DIR / 'scripts/export_text_corpus.py'),
        '--config', str(CONFIG), '--seed', '42',
    ], env=WORKER_ENV)
else:
    print('手順11は本番の選抜済み30万文対向けです。')
""")
md("""## 再開と保存

切断後は手順0〜4で同じDriveと設定を読み込み、続きの工程を実行します。`--resume` が完了済みのチェックポイントを検証して再利用します。ランタイム終了前にDriveの同期を待ってください。

成果物はDriveで管理し、ノートブックを共有する際は **編集 → すべての出力を消去** を実行してください。
""")

for i, cell in enumerate(cells):
    cell["id"] = f"cell-{i:02d}"

nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": []},
    },
)
dest = ROOT / "notebooks/jparacrawl_filtering_colab.ipynb"
dest.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, dest)
print(dest)
