# Local verification

## Text corpus export

Verification used synthetic data on a Windows CPU environment. The exporter checks `training_text` and the source fields against the production configuration and selected Parquet when it runs.

- All 128 pytest tests passed on both Python 3.11 and 3.12. Ruff lint and format checks and static notebook validation also passed.
- Two standalone exports of 300,000 synthetic pairs each produced 298,000 train, 2,000 dev, and zero test pairs. The dev bin counts were 40 / 300 / 800 / 660 / 200. Both exports had the same SHA256 hash, and the input Parquet retained its original hash, size, and modification time.
- Unit tests covered largest-remainder allocation and ties, reproducibility with the same seed, dev membership changes with a different seed, ID preservation and uniqueness, rejection of text leakage and empty text, configured text fields, and additional metadata.
- The exporter worked with the schema, export manifest, and select generation produced by the pipeline on synthetic data. It stopped when the configuration's `training_text` differed from the checkpoint.
- The CLI ran in a subprocess without importing the pipeline package. Both successful exports and validation failures preserved existing selected data, audit files, and configuration. Export writes were confined to the text corpus directory.
- The export change was scoped to the standalone exporter, its tests, and documentation.

## Lightweight statistics (2026-10-30)

All 113 pytest tests passed on both Python 3.11 and 3.12. Ruff and notebook schema, cell syntax, and output-free checks also passed. These were local tests with synthetic data.

- Generated statistics from a score manifest representing 16,668,661 rows. The fixture omitted Parquet files for the large tail with zero accepted rows and for the normalize, rule-filter, speech-filter, exact-dedup, scoring, and select stages.
- Verified that the script opened only score and dedup parts containing accepted rows and the selected export. It read four statistics columns in batches, plus `accepted` for candidates, and excluded rejected rows from mixed parts.
- Verified manifest counts, totals for multiple rejection reasons, length-bin distributions for all three groups, quotas for a 300,000-pair target, exact quantiles for selected pairs, and all PNG outputs.
- Verified checkpoint identity and selected-export compatibility with a synthetic checkpoint tree produced by the pipeline. The statistics script ran in a subprocess without importing the pipeline package.
- Verified that the completion manifest was written last, a failed overwrite left no completion marker, and resume validated output checksums without opening Parquet files.
- Verified that configuration mismatches, incomplete generations, mixed upstream generations or exports, and reads above 1.1 million physical rows were rejected before scanning data. Source changes during a run also prevented completion.
- Checked that file contents and modification times in the data, checkpoints, selected, audit, models, and configuration paths were preserved on success and failure. `git diff 74a0029 -- src scripts/bootstrap_colab.py configs/default.yaml uv.lock` showed no changes.
- Verified that notebook step 10 invoked the dedicated statistics script through the worker environment.

The user reported 16,668,661 pairs processed during production scoring, 1,000,000 score candidates, 977,822 pairs accepted after deduplication, 300,000 selected pairs, and a completed audit. The local verification used only synthetic data.

## Earlier pipeline verification (2026-10-29)

Tests ran on a Windows CPU environment using synthetic sentence pairs.

| Check | Result |
| --- | --- |
| Python 3.11 | Created an isolated environment from `uv.lock` and passed 99 pytest tests |
| Python 3.12 | Installed the package in editable mode with pip and passed 99 pytest tests |
| Python 3.13 host | Ran bootstrap from Python 3.13.14 on Windows and installed locked CPU dependencies in an isolated Python 3.11.18 environment |
| Notebook environment isolation | Checked that the worker handles stage commands, configuration, GPU checks, and pip fallback, while the notebook kernel stays free of pipeline and YAML imports |
| Colab run modes | Checked path validation for smoke and production, preservation of existing configuration, rejection of stale settings after a mode switch, and preflight checks before long-running stages |
| Colab sampling | Checked head sampling, reproducible fixed-seed reservoir sampling, preservation of manually prepared samples, and explicit rebuilds when sampling settings change |
| Colab live logs | Used mocks and real subprocesses to check streamed stdout and stderr, errors for nonzero exit codes, and scoring progress lines |
| GPU bootstrap and preflight | Mocked `nvidia-smi` to test dependency selection based on GPU availability and failure when any of the four required CUDA checks was missing |
| Synthetic smoke isolation | Ran normalize → rule-filter → speech-filter → exact-dedup with smoke settings and verified that production inputs, configuration, and outputs were preserved |
| Python 3.13 notebook helpers | Imported `colab_ops` on Python 3.13.18 and verified that it did not load the pipeline, PyYAML, fastText, TensorFlow, or PyArrow into the kernel |
| Synthetic end-to-end run | Completed normalize → rule filtering → speech filtering → exact pair deduplication → scoring with a test double → one-side deduplication → selection → audit → statistics |
| Resume | Reused completed generations and saved inference shards from an interrupted run. Verified identical outputs |
| Configuration changes and force | Reused inference after changing target or candidate counts. Rebuilt downstream stages after forcing an upstream stage |
| Corrupt checkpoint | Stopped on a SHA256 mismatch. Forcing the stage created a new generation |
| CPU language identification | Ran the official `lid.176.ftz` model with `fasttext-wheel` on 16 original synthetic pairs. Eight passed rule filtering, seven passed speech filtering, and six passed exact pair deduplication. A rerun skipped completed work |
| 300k selection | Selected 300,000 pairs from 310,000 generator-supplied pairs using SQLite without loading the full input into a list |
| Bicleaner CLI | Verified required arguments and entry point against the official PyPI wheel for version 3.6 |
| GPU, out-of-memory handling, and CPU fallback | Used a subprocess test double to verify arguments, environment variables, batch size halving, CPU retries in `auto` mode, and stopping in `cuda` mode |
| Exact deduplication before inference | Removed exact matches and passed both translations of a one-side match to scoring. This freed candidate slots and let the higher-scoring translation survive downstream selection |
| Exact deduplication resume | Rebuilt the SQLite set of seen pairs from the input, reused saved shards, and detected duplicates across shard boundaries |
| CPU imports | Imported the CLI without loading TensorFlow, PyTorch, or fastText |
| Notebook | Validated the nbformat schema and Python cell syntax and checked that no outputs were saved |
| Figures | Generated five PNGs and visually inspected labels and margins in the score, rejection reason, and selection comparison plots |
| Git ignore rules | Verified that data, outputs, audit files, checkpoints, models, logs, TSV, and Parquet files were ignored |
| Package build | Built an sdist and wheel and verified that the sdist excluded real data, models, caches, and verification outputs |
| Distributed input format | Streamed the official NTT v3 archive and verified that the first record had five columns, with English in column 3 and Japanese in column 4 using zero-based indexing. The text was neither saved nor displayed |

Synthetic demo scores came from a test double. Real Bicleaner CPU and GPU inference, NVIDIA L4 throughput and VRAM usage, Colab Drive mounting and synchronization after a disconnect, and selection of 300,000 pairs from the full corpus are outside the scope of these local checks. Use the Colab notebook to validate those runs. Its repository URL is set to `Yaaamashiro/jparacrawl-filtering`.

## User-reported Colab smoke result

The following counts were reported by the user for a Colab smoke run on real data.

| Stage | Input / accepted | Rejected |
| --- | ---: | ---: |
| Input / normalize | 50,000 | 0 |
| Rule-filter | 38,247 | 11,753 |
| Speech-filter | 36,536 | 1,711 |
| Exact normalized pair dedup | 21,923 | 14,613 |
| Bicleaner processed | 21,923 | — |
| Top candidates | 10,000 | 11,923 |

The reported run used an NVIDIA L4 with 23,034 MiB of VRAM and Bicleaner-AI 3.6. GPU scoring reached approximately 133 pairs per second and exited with code 0. The exact duplicate count reflects this input. The operational changes were outside the deduplication rules, pipeline source, existing bootstrap, default configuration, and dependency lockfile. The reported data and checkpoints were managed in the user's Colab and Drive environment.

## Reproduce the local checks

```bash
uv sync --locked --extra lid --extra dev
uv run pytest -q
uv run ruff check src scripts tests
uv run python scripts/validate_notebook.py
uv run python scripts/synthetic_demo.py
uv run python scripts/synthetic_demo.py --resume
uv run python scripts/benchmark_selection.py
```
