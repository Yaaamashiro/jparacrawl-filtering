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
print('Python:', sys.version.split()[0])
print('Commit:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
""")
md(
    "## 3. Worker environment (Python 3.11 / uv with pip fallback)\nカーネルのPythonを変更せず、NumPy<2が必要なBicleanerとfastTextを独立環境へ導入します。pip fallbackも同じ処理用環境で実行します。"
)
code("""
USE_CUDA_DEPS = False  # True: install bicleaner-ai[and-cuda] rather than using Colab CUDA.
extras = 'lid,cuda' if USE_CUDA_DEPS else 'lid,scoring'
subprocess.run([sys.executable, 'scripts/bootstrap_colab.py', '--extras', extras], check=True)
import json
runtime = json.loads((REPO_DIR / '.work/colab_environment.json').read_text())
PIPELINE_PYTHON = runtime['python']
# Only subprocesses use the worker environment; the notebook kernel stays intact.
WORKER_ENV = os.environ.copy()
for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'):
    WORKER_ENV.pop(key, None)
WORKER_ENV['PATH'] = str(Path(PIPELINE_PYTHON).parent) + os.pathsep + WORKER_ENV.get('PATH', '')
print('Notebook kernel:', sys.version.split()[0])
print('Pipeline worker:', runtime['version'], PIPELINE_PYTHON)
""")
md(
    "## 4. Config verification / language model\n既存のDrive configは上書きしません。設定変更は下のセルかDrive上のYAMLで行ってください。"
)
code("""
WORK = Path('/content/drive/MyDrive/jparacrawl-filtering')
CONFIG = WORK / 'config.yaml'
INPUT = Path('/content/drive/MyDrive/jparacrawl/v3/en-ja/en-ja.bicleaner05.txt')
config = json.loads(subprocess.check_output([
    PIPELINE_PYTHON, 'scripts/prepare_colab_config.py',
    '--defaults', 'configs/default.yaml', '--config', str(CONFIG),
    '--input', str(INPUT), '--work-dir', str(WORK),
], text=True, env=WORKER_ENV))
print(CONFIG.read_text())
def run_stage(stage, *options):
    subprocess.run([PIPELINE_PYTHON, '-m', 'jparacrawl_filtering', stage,
                    '--config', str(CONFIG), *options], check=True, env=WORKER_ENV)
run_stage('validate-config')
if not Path(config['paths']['input']).is_file():
    raise FileNotFoundError('Prepare the extracted JParaCrawl TSV on Drive first.')
run_stage('download-lid')
""")
md(
    "## 5. CPU filtering\n実行中にGPUは不要です。完了後にランタイムをGPUへ変更してもDriveから再開できます。"
)
code("run_stage('run', '--resume', '--until', 'exact-dedup')")
md(
    "## 6. GPU availability\nCUDA / TensorFlowはscoring子プロセスで使用します。configのdeviceがautoなら警告してCPUへfallbackし、cudaならGPUを必須とします。"
)
code("""
import warnings
gpu_available = json.loads(subprocess.check_output([
    PIPELINE_PYTHON, '-c',
    'import json; from jparacrawl_filtering.scoring import cuda_available; print(json.dumps(cuda_available()))',
], text=True, env=WORKER_ENV))
if gpu_available:
    subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], check=True)
else:
    if config['bicleaner']['device'] == 'cuda':
        raise RuntimeError('device=cuda requires a GPU. Select a GPU runtime before scoring.')
    warnings.warn('GPU unavailable. With device=auto, scoring falls back to CPU and may be slow.')
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
