# JParaCrawl Filtering

Build a Japanese–English training corpus for Direct Speech-to-Speech Translation research from JParaCrawl v3. The pipeline filters for translation quality and natural, sentence-like text, then selects **300,000 sentence pairs**, primarily with Japanese sentences of 30–99 characters.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yaaamashiro/jparacrawl-filtering/blob/main/notebooks/jparacrawl_filtering_colab.ipynb)

- **Selection** — Normalization and rule filtering → exact pair deduplication → Bicleaner-AI → one-side deduplication → length-bin selection
- **Resume** — Continue from checkpoints stored on Google Drive
- **Outputs** — Selected Parquet, audit CSVs, statistical reports, and training JSONL

## Using Google Colab

1. Download the **English–Japanese training set** from JParaCrawl v3.0. [Official NTT download page](https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/)
2. Extract the archive and place `en-ja.bicleaner05.txt` in `MyDrive/jparacrawl/v3/en-ja/` on Google Drive.
3. Open the notebook using the badge above and follow the steps for your task. Step numbers match the notebook headings.

| Task | Notebook steps |
| --- | --- |
| First production run | 0–4b for setup → 5 for CPU filtering → 6–7 for GPU checks and scoring → 8 for selection |
| Small smoke test | Set `RUN_MODE = 'smoke'` in step 0, then run steps 1–8 |
| Reconnect or switch to a GPU runtime | Run steps 0–4 with the same configuration, then continue. Use steps 6–7 to resume scoring |
| Inspect selection results | 9 for audit samples, 10 for lightweight statistics |
| **Export selected Parquet to JSONL** | **0–4 → 11**. Works on a CPU runtime |

Production scoring targets an NVIDIA L4. The notebook sets up a separate Python 3.11 worker environment. The default run mode is `production`. Smoke mode uses separate input and work directories and selects up to 5,000 pairs from the first 50,000 input lines by default.

Production configuration and outputs are stored in `MyDrive/jparacrawl-filtering/`. After changing run modes, rerun step 4 to select the configuration. Input formats, settings, and resume behavior are covered in the operational reference, currently in Japanese. [Operational reference](docs/reference.md)

## Outputs

| Path relative to the work directory | Contents |
| --- | --- |
| `outputs/selected/selected.parquet` | Selected pairs with raw and normalized text, IDs, and scores |
| `outputs/audit/*.csv` | Accepted and rejected samples for manual review |
| `outputs/stats/report.md` | Stage retention, rejection reasons, and length and score distributions. CSV, JSON, and PNG files are saved alongside the report |
| `outputs/text-corpus/manifest.jsonl` | Training pairs with `train` / `dev` assignments |
| `outputs/text-corpus/summary.json` | Counts, text field names, source and output SHA256 hashes, and split counts by length bin |

### Training JSONL

Step 11 splits the selected 300,000 pairs into **298,000 training pairs and 2,000 development pairs**, using seed `42` by default. The development split is stratified by length bin, and each pair retains its original `pair_id`. The `training_text` setting determines whether raw or normalized text is exported. Identical input, settings, and seed produce identical JSONL.

```json
{"pair_id":"p000012345678","split":"train","ja_text":"これは例文です。","en_text":"This is an example.","corpus":"jparacrawl_v3_filtered","length_bin":2,"bicleaner_score":0.998,"source_line":12345678}
```

## Local usage

Use Python 3.11 or 3.12 with `uv`. CPU stages and the synthetic demo support Windows and Linux. To use Bicleaner-AI, add `--extra scoring` on Linux or Colab.

```bash
git clone https://github.com/Yaaamashiro/jparacrawl-filtering.git
cd jparacrawl-filtering
uv sync --locked --extra lid --extra dev
uv run python scripts/synthetic_demo.py
```

For a real-data run, copy the default configuration and adjust the input, work directory, and model paths for your environment. The following commands assume that your copy is named `config.yaml`. [Default configuration](configs/default.yaml)

```bash
uv run python -m jparacrawl_filtering validate-config --config config.yaml
uv run python -m jparacrawl_filtering download-lid --config config.yaml
uv run python -m jparacrawl_filtering run --config config.yaml --resume --until exact-dedup
```

To generate statistics or JSONL from completed selection results, use the existing configuration file.

```bash
uv run python scripts/generate_lightweight_stats.py --config config.yaml --resume
uv run python scripts/export_text_corpus.py --config config.yaml --seed 42
```

See the operational reference for configuration, CLI commands, and checkpoint behavior. [Operational reference (Japanese)](docs/reference.md)

## Development

```bash
uv run pytest -q
uv run ruff check src scripts tests
uv run python scripts/build_notebook.py
uv run python scripts/validate_notebook.py
```

Edit the notebook generator and regenerate the Colab notebook with the commands above. Local validation scope and results are documented separately. [Notebook generator](scripts/build_notebook.py) · [Verification notes (Japanese)](docs/verification.md)

## License and data

The software is licensed under the MIT License. [LICENSE](LICENSE)

When using JParaCrawl or derived corpora, review the distribution terms and the rights associated with the original web content. Keep source data, derived corpora, audit CSVs, and checkpoints in your own environment. Models are subject to their respective terms. [Official JParaCrawl page](https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/)
