# Operational reference

## Input and configuration

The JParaCrawl v3 Japanese–English TSV has five columns and no header. The columns are the English domain, Japanese domain, original Bicleaner score, English text, and Japanese text, in that order. The default zero-based mapping is `{ja: 4, en: 3}`. The pipeline scores pairs with Bicleaner-AI independently of the supplied scores.

Use the extracted TSV or a gzip-compressed copy of that TSV. Keep `quoting: false` for the distributed data and use `quoting: true` for TSV files with CSV-style quoting. For input with a header, set `header: true` and specify column names.

| Setting | production | smoke |
| --- | --- | --- |
| Input | `/content/drive/MyDrive/jparacrawl/v3/en-ja/en-ja.bicleaner05.txt` | `/content/drive/MyDrive/jparacrawl/smoke/en-ja-smoke.txt` |
| Configuration file | `/content/drive/MyDrive/jparacrawl-filtering/config.yaml` | `/content/jparacrawl-filtering/configs/smoke.yaml` |
| Work directory | `/content/drive/MyDrive/jparacrawl-filtering` | `/content/drive/MyDrive/jparacrawl-filtering-smoke` |
| Scratch directory | `/content/jparacrawl-filtering-scratch` | `/content/jparacrawl-filtering-smoke-scratch` |
| Default target size / candidate limit / inference shard size | 300,000 / 1,000,000 / 100,000 | 5,000 / 10,000 / 5,000 |

The notebook verifies that the paths match the run mode and reuses the existing YAML configuration. A new production configuration uses `device: cuda`. In smoke mode, `head` takes the first rows, while `random` scans the entire input and samples with a fixed seed. Set `REBUILD_SMOKE_INPUT = True` to rebuild an existing smoke sample.

The local CLI resolves relative paths from the working directory. It identifies input files by path, size, and modification time. Set `input.sha256` to verify the hash of the entire input at startup. [Default configuration](../configs/default.yaml)

## Filtering and length quotas

1. Apply NFKC normalization, standardize whitespace and line breaks, and remove invisible characters. Keep a copy of the original text.
2. Check the language with fastText and apply rules for sentence length, character-count ratios, digit multisets, URLs, HTML, navigation text, and boilerplate.
3. Use sentence-ending punctuation, closing brackets and quotation marks, bullet markers, and symbol ratios to select sentences suitable for speech.
4. For pairs with identical normalized text on both sides, keep the first in input order. Score the remaining pairs with Bicleaner-AI and rank candidates by score.
5. Select candidates in score order and exclude a pair if its Japanese or English side has already been selected.
6. Select the highest-scoring pairs in each Japanese character-count bin. The default quotas are 2% for 0–9 characters, 15% for 10–29, 40% for 30–59, 33% for 60–99, and 10% for 100 or more. Fill slots left by rounding or insufficient candidates with the highest-scoring pairs across eligible bins. Record any remaining shortfall.

## Bicleaner-AI

The default model is `bitextor/bicleaner-ai-full-en-ja`. The Bicleaner CLI receives the English and Japanese text in that order and scores pairs with `--score_only --disable_hardrules`. [Model](https://huggingface.co/bitextor/bicleaner-ai-full-en-ja) · [CLI](https://github.com/bitextor/bicleaner-ai)

| `device` | Behavior |
| --- | --- |
| `cuda` | Require a GPU and stop if inference fails |
| `auto` | Try the GPU and fall back to the CPU with a warning if the GPU is unavailable or fails |
| `cpu` | Score on the CPU |

When GPU memory runs out, the pipeline halves the batch size and retries. It loads the model for each inference shard and reports progress and estimated time remaining. In Colab, step 3 selects `lid,cuda` or `lid,scoring` based on GPU availability and creates a Python 3.11 worker environment separate from the notebook kernel. Step 6 checks the worker and the Bicleaner CLI.

For reproducible experiments, download a model snapshot at a specific revision, set `bicleaner.model` to the downloaded path, and enable `offline: true`.

```bash
uv run python scripts/prepare_bicleaner_model.py \
  --revision '<commit-sha>' \
  --destination /content/drive/MyDrive/jparacrawl-filtering/models/bicleaner
```

## CLI and checkpoints

```bash
uv run python -m jparacrawl_filtering --help
uv run python -m jparacrawl_filtering score --config config.yaml --resume
uv run python -m jparacrawl_filtering select --config config.yaml --resume
uv run python -m jparacrawl_filtering audit --config config.yaml --resume
```

A stage command checks upstream results and runs any required stages. `run --until exact-dedup` stops before scoring. After completing a production run, use `scripts/generate_lightweight_stats.py` for statistics. The pipeline CLI's `stats` command and a full `run` rescan large amounts of data to compute the full distributions.

Checkpoints live in `data/<stage>/<generation>/`. `CURRENT.json` points to the completed generation and `ACTIVE.json` points to the active generation, which may be incomplete. `--resume` validates and reuses completed shards. A change to the input, relevant settings, implementation, model information, or upstream generations creates a new generation from the affected stage onward. Changing the candidate limit or final target size can reuse inference results. `--force speech-filter --resume` recomputes the specified stage and its downstream stages.

After an interruption, the pipeline rebuilds local SQLite indexes and the set of pairs seen during exact deduplication. With gzip input, it reads from the beginning to reach the unfinished portion. Allow several times the input size in Drive storage for intermediate Parquet files and retained generations. Use one runtime per work directory and wait for Drive to finish syncing before ending the session.

## Review and export after selection

- **Audit** — Sample 500 accepted and 200 rejected pairs with a fixed seed. Review translation alignment, web fragments, and suitability for speech in the CSV files under `outputs/audit/`.
- **Lightweight statistics** — Read stage counts and rejection reasons from manifests and project only the required columns from the candidate, deduplicated candidate, and selected data. Distributions cover those three groups. For `score` and `dedup`, the script reads only parts that contain accepted rows and stops before reading if those parts exceed 1.1 million rows in either stage. See `stats.json` under `io` for actual row counts, `length_bins.csv` for length comparisons, and `report.md` for the summary. A pair can have several rejection reasons, so reason counts can differ from the number of rejected pairs. `--resume` reuses a completed report when the inputs and output checksums match.
- **Text corpus** — Use `ja_norm` / `en_norm` when `training_text` is `normalized` and `ja_raw` / `en_raw` when it is `raw`. The exporter checks the configuration against selection metadata and validates ID uniqueness, nonempty text, and train/dev overlap. It allocates dev counts using the largest remainder method based on observed bin counts. Within each bin, it selects by a SHA256 rank based on the seed, bin ID, and original pair ID. The defaults expect 300,000 pairs. For a small test, pass `--expected-total 20 --dev-size 4`. The exporter writes `summary.json` after saving the JSONL.

The selected Parquet contains source row numbers, raw and normalized text, the training text fields `ja` and `en`, filtering features, and scores. The JSONL export records the source schema in `summary.json`.
