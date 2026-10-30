"""Export the completed selected Parquet into a reproducible text-only train/dev JSONL."""

import argparse
import hashlib
import json
import math
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

CORPUS = "jparacrawl_v3_filtered"
SPLIT_METHOD = "largest_remainder_then_sha256_seed_bin_pair_id_v1"


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_identity(manifest):
    value = {k: manifest[k] for k in ("signature", "generation", "parts")}
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def allocate_dev(counts, dev_size):
    """Integer arithmetic keeps quotas exact and ties deterministic (ascending bin ID)."""
    total = sum(counts.values())
    if total < 1 or dev_size < 0 or dev_size > total:
        raise ValueError("Require a nonempty corpus and 0 <= dev_size <= total")
    if any(not isinstance(b, int) or b < 0 or n < 0 for b, n in counts.items()):
        raise ValueError("Invalid length-bin counts")
    quotas = {b: n * dev_size // total for b, n in sorted(counts.items())}
    remaining = dev_size - sum(quotas.values())
    ordered = sorted(counts, key=lambda b: (-(counts[b] * dev_size % total), b))
    for b in ordered[:remaining]:
        quotas[b] += 1
    return quotas


def safe_child(directory, name):
    path = directory / name
    if path.resolve().parent != directory.resolve() or path.is_symlink():
        raise ValueError(f"Unsafe export path: {path}")
    return path


def inspect_source(config, root, source):
    training_text = config.get("training_text")
    if training_text not in ("raw", "normalized"):
        raise ValueError("Config training_text must explicitly be raw or normalized")
    suffix = "raw" if training_text == "raw" else "norm"
    fields = {"ja": "ja_" + suffix, "en": "en_" + suffix}
    with pq.ParquetFile(source) as parquet:
        schema = parquet.schema_arrow
        rows = parquet.metadata.num_rows
    for name in ("pair_id", fields["ja"], fields["en"]):
        if name not in schema.names or not (
            pa.types.is_string(schema.field(name).type)
            or pa.types.is_large_string(schema.field(name).type)
        ):
            raise ValueError(f"Missing or non-string source field: {name}")
    if "length_bin" not in schema.names or not pa.types.is_integer(schema.field("length_bin").type):
        raise ValueError("Source must contain integer length_bin")
    if ("ja" in schema.names) != ("en" in schema.names):
        raise ValueError("Source training aliases ja/en must both be present")
    provenance = {
        "training_text": training_text,
        "config_checked": True,
        "training_aliases_checked": "ja" in schema.names,
    }
    exported_path = source.parent / "manifest.json"
    exported = read_json(exported_path) if exported_path.exists() else None
    if exported is not None and exported["rows"] != rows:
        raise ValueError("Selected export manifest row count differs from Parquet")
    current_path = root / "data/select/CURRENT.json"
    if current_path.exists():
        current = read_json(current_path)
        generation = safe_child(current_path.parent, current["generation"])
        manifest = read_json(generation / "manifest.json")
        if (
            manifest.get("complete") is not True
            or manifest["generation"] != current["generation"]
            or manifest["signature"] != current["signature"]
            or manifest["stage"] != "select"
        ):
            raise ValueError("Select CURRENT must reference a matching complete generation")
        if manifest["settings"]["training_text"] != training_text:
            raise ValueError("Config training_text differs from completed select checkpoint")
        if exported is None or exported["upstream"] != checkpoint_identity(manifest):
            raise ValueError("Selected export belongs to a different select checkpoint")
        if sum(p["accepted"] for p in manifest["parts"]) != rows:
            raise ValueError("Select checkpoint count differs from selected export")
        provenance["select_generation"] = current["generation"]
        provenance["select_identity"] = exported["upstream"]
    return schema, rows, fields, provenance, exported


def load_pairs(db, source, schema, fields, seed):
    columns = ["pair_id", fields["ja"], fields["en"], "length_bin"]
    optional = [name for name in ("bicleaner_score", "source_line") if name in schema.names]
    aliases = ["ja", "en"] if "ja" in schema.names else []
    accepted_column = ["accepted"] if "accepted" in schema.names else []
    counts = Counter()
    db.execute("PRAGMA cache_size=-32768")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("""CREATE TABLE pairs (
        position INTEGER PRIMARY KEY, pair_id TEXT NOT NULL UNIQUE,
        ja TEXT NOT NULL, en TEXT NOT NULL, bin INTEGER NOT NULL,
        score REAL, source_line INTEGER, rank TEXT NOT NULL,
        split TEXT NOT NULL DEFAULT 'train')""")
    position = 0
    with pq.ParquetFile(source) as parquet:
        for batch in parquet.iter_batches(
            batch_size=8192, columns=columns + optional + aliases + accepted_column
        ):
            records = []
            for row in batch.to_pylist():
                pair_id = row["pair_id"]
                if not isinstance(pair_id, str) or not pair_id.strip():
                    raise ValueError(f"Empty pair_id at source row {position + 1}")
                ja, en = row[fields["ja"]], row[fields["en"]]
                if any(not isinstance(t, str) or not t.strip() for t in (ja, en)):
                    raise ValueError(f"Empty training text at source row {position + 1}")
                if aliases and (row["ja"] != ja or row["en"] != en):
                    raise ValueError("Source ja/en aliases disagree with config training_text")
                if accepted_column and row["accepted"] is not True:
                    raise ValueError("Selected export contains an unaccepted row")
                b = row["length_bin"]
                if not isinstance(b, int) or b < 0:
                    raise ValueError(f"Invalid length_bin at source row {position + 1}")
                score = row.get("bicleaner_score")
                if "bicleaner_score" in optional and (
                    score is None or not math.isfinite(score) or not 0 <= score <= 1
                ):
                    raise ValueError(f"Invalid bicleaner_score at source row {position + 1}")
                source_line = row.get("source_line")
                if source_line is not None and (
                    not isinstance(source_line, int) or source_line < 1
                ):
                    raise ValueError(f"Invalid source_line at source row {position + 1}")
                rank = hashlib.sha256(encode([seed, b, pair_id]).encode("utf-8")).hexdigest()
                records.append((position, pair_id, ja, en, b, score, source_line, rank))
                counts[b] += 1
                position += 1
            try:
                db.executemany(
                    """INSERT INTO pairs
                    (position,pair_id,ja,en,bin,score,source_line,rank) VALUES (?,?,?,?,?,?,?,?)""",
                    records,
                )
            except sqlite3.IntegrityError as error:
                raise ValueError("Duplicate pair_id in selected Parquet") from error
            print(f"Reading selected corpus: {position:,} rows", flush=True)
    db.execute("CREATE INDEX stratified_order ON pairs(bin,rank,pair_id)")
    db.commit()
    return counts, optional


def split_pairs(db, counts, dev_size):
    quotas = allocate_dev(counts, dev_size)
    for b, quota in quotas.items():
        db.execute(
            """UPDATE pairs SET split='dev' WHERE position IN (
            SELECT position FROM pairs WHERE bin=? ORDER BY rank,pair_id LIMIT ?)""",
            (b, quota),
        )
    # Exact text comparison, not a text hash: detect leakage without collision assumptions.
    leakage = db.execute("""SELECT 1 FROM pairs GROUP BY ja,en
        HAVING COUNT(DISTINCT split)>1 LIMIT 1""").fetchone()
    if leakage:
        raise ValueError("Identical (ja_text, en_text) occurs in train and dev; export aborted")
    actual = dict(db.execute("SELECT bin,COUNT(*) FROM pairs WHERE split='dev' GROUP BY bin"))
    if any(actual.get(b, 0) != n for b, n in quotas.items()):
        raise ValueError("Dev stratification validation failed")
    split_counts = dict(db.execute("SELECT split,COUNT(*) FROM pairs GROUP BY split"))
    if (
        split_counts.get("dev", 0) != dev_size
        or split_counts.get("train", 0) != sum(counts.values()) - dev_size
    ):
        raise ValueError("Train/dev size validation failed")
    return quotas


def export(config_path, expected_total=300000, dev_size=2000, seed=42):
    for name, value in (("expected_total", expected_total), ("dev_size", dev_size), ("seed", seed)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if expected_total < 1 or dev_size > expected_total:
        raise ValueError("Require expected_total > 0 and dev_size <= expected_total")
    config_path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = Path(config["paths"]["work_dir"]).expanduser().resolve()
    source = root / "outputs/selected/selected.parquet"
    out = root / "outputs/text-corpus"
    if out.resolve() != root / "outputs/text-corpus":
        raise ValueError("Text corpus output must not redirect to another directory")
    schema, rows, fields, provenance, exported = inspect_source(config, root, source)
    if rows != expected_total:
        raise ValueError(
            f"Expected {expected_total:,} selected pairs, found {rows:,}; no export written"
        )
    before = source.stat()
    print(
        f"=== Text corpus export ===\nSource: {source}\nSelected pairs: {rows:,}\n"
        f"Training text: {config['training_text']} ({fields['ja']}, {fields['en']})\nSeed: {seed}",
        flush=True,
    )
    print("Hashing selected Parquet...", flush=True)
    source_sha256 = file_hash(source)
    if exported is not None and exported["sha256"] != source_sha256:
        raise ValueError("Selected Parquet SHA256 differs from its export manifest")
    out.mkdir(parents=True, exist_ok=True)
    output = safe_child(out, "manifest.jsonl")
    summary_path = safe_child(out, "summary.json")
    temporary = summary_temp = None
    try:
        # Keep SQLite on local temp storage, avoiding random I/O against mounted Drive.
        with tempfile.TemporaryDirectory(prefix="jparacrawl-text-export-") as scratch:
            db = sqlite3.connect(str(Path(scratch) / "pairs.sqlite"))
            try:
                counts, optional = load_pairs(db, source, schema, fields, seed)
                if sum(counts.values()) != expected_total:
                    raise ValueError("Selected row count changed while reading")
                quotas = split_pairs(db, counts, dev_size)
                print(
                    f"Split validated: train={rows - dev_size:,}, dev={dev_size:,}, test=0",
                    flush=True,
                )
                checksum = hashlib.sha256()
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=out, prefix="manifest.", suffix=".partial", delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    for pair_id, ja, en, b, score, source_line, split in db.execute(
                        "SELECT pair_id,ja,en,bin,score,source_line,split FROM pairs ORDER BY position"
                    ):
                        record = {
                            "pair_id": pair_id,
                            "split": split,
                            "ja_text": ja,
                            "en_text": en,
                            "corpus": CORPUS,
                            "length_bin": b,
                        }
                        if "bicleaner_score" in optional:
                            record["bicleaner_score"] = score
                        if "source_line" in optional:
                            record["source_line"] = source_line
                        line = (encode(record) + "\n").encode("utf-8")
                        stream.write(line)
                        checksum.update(line)
            finally:
                db.close()
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Selected Parquet changed during export; export aborted")
        if yaml.safe_load(config_path.read_text(encoding="utf-8")) != config:
            raise RuntimeError("Config changed during export; export aborted")
        summary = {
            "complete": True,
            "corpus": CORPUS,
            "source": str(source),
            "source_sha256": source_sha256,
            "output_sha256": checksum.hexdigest(),
            "seed": seed,
            "split_method": SPLIT_METHOD,
            "total_pairs": rows,
            "train_pairs": rows - dev_size,
            "dev_pairs": dev_size,
            "test_pairs": 0,
            "text_fields": fields,
            "training_text": config["training_text"],
            "config": str(config_path),
            "provenance": provenance,
            "source_schema": [{"name": field.name, "type": str(field.type)} for field in schema],
            "length_bin_counts": {
                "all": {str(b): counts[b] for b in sorted(counts)},
                "train": {str(b): counts[b] - quotas[b] for b in sorted(counts)},
                "dev": {str(b): quotas[b] for b in sorted(counts)},
            },
            "validation": {
                "unique_pair_ids": True,
                "train_dev_pair_id_overlap": 0,
                "train_dev_text_pair_overlap": 0,
                "nonempty_text": True,
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=out,
            prefix="summary.",
            suffix=".partial",
            delete=False,
        ) as stream:
            summary_temp = Path(stream.name)
            json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        # summary.json is the completion marker; publish it only after the corpus.
        summary_path.unlink(missing_ok=True)
        temporary.replace(output)
        summary_temp.replace(summary_path)
        print(
            f"Text corpus export complete.\nOutput: {output}\nSummary: {summary_path}", flush=True
        )
        return summary
    finally:
        for path in (temporary, summary_temp):
            if path is not None:
                path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-total", type=int, default=300000)
    parser.add_argument("--dev-size", type=int, default=2000)
    args = parser.parse_args()
    export(args.config, expected_total=args.expected_total, dev_size=args.dev_size, seed=args.seed)


if __name__ == "__main__":
    main()
