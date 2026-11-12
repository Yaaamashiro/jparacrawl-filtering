"""Regenerate the output-free Colab notebook from reviewable Python cell sources."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""# Filter Japanese–English sentence pairs from JParaCrawl

Select sentence pairs suitable for speech from JParaCrawl v3 and save the Parquet and training JSONL files to Google Drive.

| Task | Notebook steps |
| --- | --- |
| First production run | **0–4b** for setup → **5** for filtering → **6–7** for GPU checks and scoring → **8** for selection |
| Small smoke test | Select `smoke` in step **0**, then run **1–8** |
| Reconnect or switch runtimes | Run **0–4** with the same configuration, then continue from the next step |
| Inspect selection results | **9** for audit samples, **10** for lightweight statistics |
| **Export selected Parquet to JSONL** | **0–4 → 11**. Works on a CPU runtime |

Before your first run, place the extracted `en-ja.bicleaner05.txt` in `MyDrive/jparacrawl/v3/en-ja/` on Google Drive. Production Bicleaner scoring targets an NVIDIA L4. You can complete step 5 on a CPU runtime before switching to a GPU.

[Repository and input data instructions](https://github.com/Yaaamashiro/jparacrawl-filtering#using-google-colab)
""")
md("""## 0. Choose a run mode

- `production` (default) uses the production configuration and outputs.
- `smoke` creates a small input sample in a separate work directory. `head` takes the first rows, and `random` samples the full input with a fixed seed.

After changing modes, rerun step 4 to select the configuration.
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
md("## 1. Mount Google Drive")
code("""
from google.colab import drive
drive.mount('/content/drive')
""")
md("""## 2. Clone or update the repository

The notebook updates an existing checkout with `git pull --ff-only`. For reproducible runs, set `REPO_REF` to a commit SHA or tag.
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
md("""## 3. Set up the worker environment

This step creates a Python 3.11 environment separate from the notebook kernel and installs dependencies based on GPU availability. Run it again after switching runtimes.
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
md("""## 4. Check the configuration and output directory

Check the displayed run mode, configuration file, input, and work directory. Production runs reuse the existing YAML on Drive. A new production configuration uses `device: cuda` and requires a GPU for scoring.
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
md("""## 4b. Prepare the input and language identification model

Run this step before filtering for the first time. In smoke mode, it samples the input. In both modes, it checks the input and configuration and prepares the fastText model.
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
md("""## 5. Filter on the CPU

Run normalization, rule and speech filtering, and exact deduplication. If you switch to an L4 afterward, run **0–4 → 6**.
""")
code("run_stage('run', '--resume', '--until', 'exact-dedup')")
md("""## 6. Check the scoring environment

Check GPU detection and whether CUDA and the Bicleaner CLI are available to the worker. With `device: cuda`, this step stops if the GPU is unavailable.
""")
code("""
config = read_active_config()
ops.validate_run(RUN_MODE, CONFIG, config, LAYOUT)
ops.gpu_preflight(PIPELINE_PYTHON, config, WORKER_ENV)
""")
md("""## 7. Score candidates with Bicleaner-AI

Score candidates and display progress and estimated time remaining. The first run downloads the model. After an interruption, run steps 0–4 and 6 before resuming here.
""")
code("run_stage('score', '--resume')")
md("""## 8. Select sentence pairs

Remove pairs with duplicate text on either side and select pairs by length bin. The output is `outputs/selected/selected.parquet` under the work directory.
""")
code("""
run_stage('select', '--resume')
# target_size is saved in the Drive config; changing it preserves inference shards.
""")
md("""## 9. Sample pairs for manual review

By default, this step saves CSV samples of 500 accepted and 200 rejected pairs under `outputs/audit/`.
""")
code("""
run_stage('audit', '--resume')
print('Review:', WORK / 'outputs/audit')
""")
md("""## 10. Summarize selection results

Use completed checkpoints to summarize stage counts and distributions. This step runs on the CPU and saves the report and figures under `outputs/stats/`.
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
print(f"Selected {summary['selected_size']:,} of {summary['target_size']:,} target pairs")
print('Report:', STATS / 'report.md')
for name in ['stage_retention.png', 'selected_ja_length.png', 'selected_en_words.png',
             'selected_bicleaner_scores.png', 'length_bin_comparison.png',
             'candidate_bicleaner_scores.png', 'reject_reasons.png']:
    display(Image(filename=str(STATS / name)))
""")
md("""## 11. Export training JSONL

Split the 300,000 selected production pairs into **298,000 train pairs and 2,000 dev pairs** using seed 42. Each pair retains its original `pair_id`. To export an existing selected Parquet, load the production configuration with steps 0–4 and run this step directly.

This step saves `outputs/text-corpus/manifest.jsonl` and `outputs/text-corpus/summary.json` to Drive.
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
    print('Step 11 expects 300,000 selected pairs from a production run.')
""")
md("""## Resume and save

After a disconnect, run steps 0–4 with the same Drive and configuration, then continue from the next step. `--resume` validates and reuses completed checkpoints. Wait for Drive to finish syncing before ending the runtime.

Keep outputs on Drive. Use **Edit → Clear all outputs** before sharing the notebook.
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
