"""Regenerate the output-free Colab notebook from reviewable Python cell sources."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""# JParaCrawl filtering for Direct S2ST

NotebookカーネルはPython 3.13でも使用できます。処理用Python 3.11は独立環境へ自動導入します。
Google Drive / optional NVIDIA L4. 原データの利用条件を公式ページで確認し、Driveへ展開してください。
このNotebookは実文の出力を保存しません。切断後は同じconfigとDriveから `--resume` で再開します。
clone先は `Yaaamashiro/jparacrawl-filtering` に設定済みです。
デフォルトはsmokeです。本番では冒頭のRUN_MODEをproductionへ変更してください。
""")
md(
    "## 0. Run mode / smoke sampling settings\nproductionとsmokeは入力・config・出力を分離します。モード変更後はconfig選択以降も再実行してください。"
)
code("""
RUN_MODE = 'smoke'  # 'smoke' or 'production'; explicitly select production for the full run.
SMOKE_SAMPLE_MODE = 'head'  # 'head': fast pipeline check; 'random': scans the full input.
SMOKE_SAMPLE_SIZE = 50000
SMOKE_SEED = 42
REBUILD_SMOKE_INPUT = False  # Preserve existing smoke data unless explicitly requested.
if RUN_MODE not in ('smoke', 'production'):
    raise ValueError("RUN_MODE must be 'smoke' or 'production'")
if SMOKE_SAMPLE_MODE not in ('head', 'random'):
    raise ValueError("SMOKE_SAMPLE_MODE must be 'head' or 'random'")
""")
md("## 1. Google Drive mount")
code("""
from google.colab import drive
drive.mount('/content/drive')
""")
md("## 2. Repository clone")
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
md(
    "## 3. Worker environment (Python 3.11 / uv with pip fallback)\nカーネルのPythonを変更せず、NumPy<2が必要なBicleanerとfastTextを独立環境へ導入します。pip fallbackも同じ処理用環境で実行します。"
)
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
md(
    "## 4. Mode-specific config\nproductionはDriveの既存configを保持します。新規本番configはdevice=cuda。smokeは専用configとtarget=5000 / candidates=10000 / shard=5000を使用します。パスがモードと矛盾すれば停止します。"
)
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
md(
    "## 4b. Smoke input / language model\nheadは先頭50,000行を高速に抽出します。randomは固定seedのreservoir samplingで全入力を走査するため時間がかかります。既存の手動smoke入力もデフォルトで保持します。"
)
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
md(
    "## 5. CPU filtering\n実行中にGPUは不要です。完了後にランタイムをGPUへ変更してもDriveから再開できます。"
)
code("run_stage('run', '--resume', '--until', 'exact-dedup')")
md(
    "## 6. GPU / Bicleaner preflight\nnvidia-smi、workerのcuda_available()、Bicleaner CLIとversionを表示します。device=cudaでGPUを使えなければ停止します。scoring起動直前にも再検査します。"
)
code("""
config = read_active_config()
ops.validate_run(RUN_MODE, CONFIG, config, LAYOUT)
ops.gpu_preflight(PIPELINE_PYTHON, config, WORKER_ENV)
""")
md(
    "## 7. Bicleaner-AI scoring\n完全一致の正規化文対は前段で除外済みです。100kごとにthroughput / ETAを表示。初回モデルdownloadには時間がかかります。L4を必須にする場合はconfigのdeviceをcudaに設定してください。再現実験ではREADMEのsnapshot固定手順を使ってください。"
)
code("run_stage('score', '--resume')")
md("## 8. Deduplication and length-aware final selection")
code("""
run_stage('select', '--resume')
# target_size is saved in the Drive config; changing it preserves inference shards.
""")
md(
    "## 9. Manual audit\n採用500件・不採用200件をDriveへ保存します。原文はNotebook出力に表示しません。"
)
code("""
run_stage('audit', '--resume')
print('Review:', Path(config['paths']['work_dir']) / 'outputs/audit')
""")
md("## 10. Statistics and figures")
code("""
run_stage('stats', '--resume')
import json
from IPython.display import Image, display
STATS = Path(config['paths']['work_dir']) / 'outputs/stats'
summary = json.loads((STATS / 'stats.json').read_text())
print(json.dumps({'stages': summary['stages'], 'shortfall': summary['shortfall']}, indent=2))
for name in ['ja_length.png', 'en_words.png', 'bicleaner_scores.png',
             'selected_vs_rejected.png', 'reject_reasons.png']:
    display(Image(filename=str(STATS / name)))
""")
md("""## Reconnect / rerun
同じDriveをmountしてconfigを再利用してください。完了stageと完了shardは検証後skipされます。
RUN_MODEを再設定し、Drive → repo → worker → config選択を再実行してください。
CPU filtering済みならGPUへ切り替えた後はGPU preflight → score --resumeから再開できます。
`--force speech-filter --resume` で該当stageと後続を再計算できます。
同じwork_dirへ複数ランタイムから同時書込しないでください。
公開前に **Edit → Clear all outputs** を実行してください。データ・監査CSVをGitへ追加しないでください。
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
