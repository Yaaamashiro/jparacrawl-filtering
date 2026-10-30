"""Read completed checkpoint metadata and bounded candidate exports; never run Pipeline."""

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

STAGES = (
    "normalize",
    "rule-filter",
    "speech-filter",
    "exact-dedup",
    "scoring",
    "score",
    "dedup",
    "select",
)
COLUMNS = ("ja_char_count", "en_word_count", "bicleaner_score", "length_bin")
# Cap physical rows, including rejected rows in a boundary part, BEFORE opening any Parquet.
MAX_SCAN_ROWS = 1_100_000
REPORT_VERSION = 1


def log(message):
    print(message, flush=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    # Same serialization as checkpoint identity; no package import or initialization.
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def child(directory, name):
    """Only read/write direct children; disallow traversal and symlink escapes."""
    directory = Path(directory).resolve()
    path = directory / name
    if path.resolve().parent != directory:
        raise ValueError(f"Unsafe child path: {path}")
    return path


def stage_metadata(root, name):
    base = root / "data" / name
    current = read_json(base / "CURRENT.json")
    generation = child(base, current["generation"])
    manifest = read_json(generation / "manifest.json")
    if (
        manifest.get("complete") is not True
        or manifest["stage"] != name
        or manifest["generation"] != current["generation"]
        or manifest["signature"] != current["signature"]
    ):
        raise ValueError(f"{name}: CURRENT must reference a matching complete generation")
    parts = manifest["parts"]
    reasons = Counter()
    for part in parts:
        child(generation, part["file"])
        if not 0 <= part["accepted"] <= part["rows"]:
            raise ValueError(f"{name}: invalid part counts")
        reasons.update(part["reasons"])
    rows = sum(p["rows"] for p in parts)
    accepted = sum(p["accepted"] for p in parts)
    identity = digest({k: manifest[k] for k in ("signature", "generation", "parts")})
    return {
        "path": generation,
        "manifest": manifest,
        "identity": identity,
        "current": current,
        "counts": {
            "stage": name,
            "input": rows,
            "accepted": accepted,
            "rejected": rows - accepted,
            "reject_rate": (rows - accepted) / max(1, rows),
            "reject_reasons": dict(reasons),
            "number_of_shards": len(parts),
            "complete": True,
            "metrics": manifest.get("metrics", {}),
        },
    }


def sources(root, config):
    metadata = {name: stage_metadata(root, name) for name in STAGES}
    for before, after in zip(STAGES, STAGES[1:]):
        if metadata[after]["manifest"]["upstream"] != metadata[before]["identity"]:
            raise ValueError(f"Mixed checkpoint generations: {before} -> {after}")
    settings = metadata["select"]["manifest"]["settings"]
    for key in ("target_size", "length_bins"):
        if settings[key] != config[key]:
            raise ValueError(f"Config {key} differs from completed select generation")
    selected = root / "outputs/selected/selected.parquet"
    exported = read_json(selected.parent / "manifest.json")
    if exported["upstream"] != metadata["select"]["identity"]:
        raise ValueError("Selected export belongs to a different select generation")
    if exported["rows"] != metadata["select"]["counts"]["accepted"]:
        raise ValueError("Selected export row count differs from select manifest")
    plans = {}
    for name in ("score", "dedup"):
        parts = [p for p in metadata[name]["manifest"]["parts"] if p["accepted"] > 0]
        physical = sum(p["rows"] for p in parts)
        if physical > MAX_SCAN_ROWS:
            raise ValueError(
                f"{name}: accepted parts require {physical:,} physical rows; "
                f"bounded scan limit is {MAX_SCAN_ROWS:,}. No Parquet was scanned."
            )
        plans[name] = parts
    if exported["rows"] > MAX_SCAN_ROWS:
        raise ValueError("Selected export exceeds bounded scan limit")
    stat = selected.stat()
    snapshot = {
        "stage_manifests": {n: digest(m["manifest"]) for n, m in metadata.items()},
        "selected_export": exported,
        "selected_file": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
    }
    return metadata, selected, plans, snapshot


def bin_label(b):
    return f"{b['min']}–{b['max']}" if b["max"] is not None else f"{b['min']}+"


class Distribution:
    def __init__(self, bins, summaries=False):
        self.bins = bins
        self.rows = 0
        self.length_bins = np.zeros(len(bins), dtype=np.int64)
        self.histograms = {
            "ja_char_count": np.zeros(81, dtype=np.int64),
            "en_word_count": np.zeros(101, dtype=np.int64),
            "bicleaner_score": np.zeros(50, dtype=np.int64),
        }
        self.values = {k: [] for k in COLUMNS[:3]} if summaries else None

    def add(self, batch, selected=False):
        arrays = {}
        for name in COLUMNS[:3]:
            column = batch.column(batch.schema.get_field_index(name))
            if column.null_count:
                raise ValueError(f"Null {name} in accepted corpus")
            values = column.to_numpy(zero_copy_only=False)
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"Invalid {name} in accepted corpus")
            arrays[name] = values
            if name == "bicleaner_score" and np.any(values > 1):
                raise ValueError("Invalid Bicleaner probability")
            width = {"ja_char_count": 5, "en_word_count": 2, "bicleaner_score": 0.02}[name]
            counts = self.histograms[name]
            index = np.minimum((values / width).astype(np.int64), len(counts) - 1)
            counts += np.bincount(index, minlength=len(counts))
            if self.values is not None:
                self.values[name].append(values.copy())
        ja = arrays["ja_char_count"]
        assigned = np.full(len(ja), -1, dtype=np.int64)
        for i, b in enumerate(self.bins):
            mask = (ja >= b["min"]) & ((ja <= b["max"]) if b["max"] is not None else True)
            if np.any(assigned[mask] != -1):
                raise ValueError("Overlapping length bins")
            assigned[mask] = i
            self.length_bins[i] += np.count_nonzero(mask)
        if np.any(assigned < 0):
            raise ValueError("Accepted corpus contains lengths outside configured bins")
        if selected:
            column = batch.column(batch.schema.get_field_index("length_bin"))
            if column.null_count or not np.array_equal(column.to_numpy(), assigned):
                raise ValueError("Selected length_bin disagrees with Japanese character count")
        self.rows += len(ja)

    def result(self):
        result = {
            "rows": self.rows,
            "exact": True,
            "length_bins": self.length_bins.tolist(),
            "histograms": {k: v.tolist() for k, v in self.histograms.items()},
        }
        if self.values is not None:
            summaries = {}
            for key, chunks in self.values.items():
                values = np.concatenate(chunks) if chunks else np.array([], dtype=float)
                if len(values):
                    q = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9], method="linear")
                    summaries[key] = {
                        "count": len(values),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "mean": float(values.mean()),
                        **dict(zip(("p10", "p25", "median", "p75", "p90"), map(float, q))),
                    }
                else:
                    summaries[key] = {
                        "count": 0,
                        **dict.fromkeys(
                            ("min", "max", "mean", "median", "p10", "p25", "p75", "p90")
                        ),
                    }
            result["summary"] = summaries
        return result


def read_distribution(paths, bins, expected, selected=False):
    distribution = Distribution(bins, summaries=selected)
    for index, (path, part) in enumerate(paths, start=1):
        log(f"  Reading {path.parent.parent.name}/{path.name} ({index}/{len(paths)})")
        with pq.ParquetFile(path) as parquet:
            if parquet.metadata.num_rows != part["rows"]:
                raise ValueError(f"Parquet row count differs from manifest: {path}")
            columns = list(COLUMNS) + ([] if selected else ["accepted"])
            kept = 0
            for batch in parquet.iter_batches(batch_size=32768, columns=columns):
                if not selected:
                    accepted = batch.column(batch.schema.get_field_index("accepted"))
                    if accepted.null_count:
                        raise ValueError("Null accepted flag")
                    batch = batch.filter(accepted)
                kept += batch.num_rows
                distribution.add(batch, selected=selected)
            if kept != part["accepted"]:
                raise ValueError(f"Accepted count differs from manifest: {path}")
    if distribution.rows != expected:
        raise ValueError("Corpus count differs from stage manifest")
    return distribution.result()


def atomic_json(out, name, value):
    path = child(out, name)
    temporary = child(out, name + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(out, name, rows, fields):
    temporary = child(out, name + ".partial")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(child(out, name))


def write_figures(out, stages, distributions, bins, reasons):
    # Set before import: a Colab inline backend is unavailable in the worker.
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(child(out, ".matplotlib"))
    import matplotlib.pyplot as plt

    files = []

    def save(fig, name):
        try:
            fig.tight_layout()
            temporary = child(out, name + ".partial")
            fig.savefig(temporary, format="png", dpi=150)
            temporary.replace(child(out, name))
            files.append(name)
        finally:
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    retention = [s for s in stages if s["stage"] != "scoring"]
    ax.bar([s["stage"] for s in retention], [s["accepted"] for s in retention])
    ax.set(ylabel="Accepted pairs")
    ax.tick_params(axis="x", rotation=25)
    save(fig, "stage_retention.png")
    for key, width, xlabel, filename in (
        (
            "ja_char_count",
            5,
            "Japanese characters (last bin includes overflow)",
            "selected_ja_length.png",
        ),
        ("en_word_count", 2, "English words (last bin includes overflow)", "selected_en_words.png"),
        ("bicleaner_score", 0.02, "Bicleaner probability", "selected_bicleaner_scores.png"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        counts = distributions["selected"]["histograms"][key]
        ax.stairs(counts, np.arange(len(counts) + 1) * width)
        ax.set(xlabel=xlabel, ylabel="Selected pairs")
        save(fig, filename)
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(bins))
    for offset, (name, data) in enumerate(distributions.items()):
        ax.bar(
            positions + (offset - 1) * 0.25,
            np.array(data["length_bins"]) / max(1, data["rows"]),
            width=0.25,
            label=name,
        )
    ax.set(xticks=positions, xticklabels=[bin_label(b) for b in bins], ylabel="Fraction of pairs")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3)
    save(fig, "length_bin_comparison.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, data in distributions.items():
        counts = np.array(data["histograms"]["bicleaner_score"])
        ax.stairs(counts / max(1, data["rows"]), np.linspace(0, 1, 51), label=name)
    ax.set(xlabel="Bicleaner probability", ylabel="Fraction of pairs")
    ax.legend()
    save(fig, "candidate_bicleaner_scores.png")
    fig, ax = plt.subplots(figsize=(8, max(3, len(reasons) * 0.26)))
    items = sorted(reasons.items(), key=lambda item: item[1])
    ax.barh([k for k, _ in items], [v for _, v in items])
    ax.set(xlabel="Reason occurrences (a pair may have several)")
    save(fig, "reject_reasons.png")
    return files


def write_report(out, report, bin_rows, figures):
    lines = [
        "# Lightweight filtering statistics",
        "",
        f"Selected **{report['selected_size']:,}** / target **{report['target_size']:,}** pairs.",
        "",
        report["omissions"]["full_scoring"],
        "",
        report["omissions"]["early_lengths"],
        "",
        "Candidate, deduplicated and selected distributions are exact.",
        "",
        "## Stage retention (manifest only)",
        "",
        "| Stage | Input | Accepted | Rejected | Reject rate | Shards | Complete |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for s in report["stages"]:
        lines.append(
            f"| {s['stage']} | {s['input']:,} | {s['accepted']:,} | {s['rejected']:,} | "
            f"{s['reject_rate']:.2%} | {s['number_of_shards']} | {s['complete']} |"
        )
    lines += [
        "",
        "## Final length quotas",
        "",
        "| Bin | Target count | Actual count | Target ratio | Actual ratio | Difference |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for b in bin_rows:
        if b["corpus"] == "selected":
            lines.append(
                f"| {b['bin']} | {b['target_count']:,} | {b['actual_count']:,} | "
                f"{b['target_ratio']:.2%} | {b['actual_ratio']:.2%} | {b['difference']:+,} |"
            )
    lines += [
        "",
        "## Candidate → dedup → selected",
        "",
        "| Corpus | Bin | Count | Ratio |",
        "| --- | --- | ---: | ---: |",
    ]
    for b in bin_rows:
        lines.append(
            f"| {b['corpus']} | {b['bin']} | {b['actual_count']:,} | {b['actual_ratio']:.2%} |"
        )
    lines += [
        "",
        "## Exact selected summaries",
        "",
        "Quantiles use linear interpolation. Empty corpora have null summaries.",
        "",
        "| Metric | Min | Max | Mean | Median | p10 | p25 | p75 | p90 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, values in report["distributions"]["selected"]["summary"].items():
        cells = [
            "—" if values[k] is None else f"{values[k]:.4g}"
            for k in ("min", "max", "mean", "median", "p10", "p25", "p75", "p90")
        ]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Reject reasons",
        "",
        "A pair may have multiple reasons; reason totals need not equal rejected pairs.",
        "",
        "| Reason | Occurrences |",
        "| --- | ---: |",
    ]
    lines += [f"| {k} | {v:,} |" for k, v in sorted(report["reject_reasons"].items())]
    lines += ["", "## Figures", ""] + [f"![{f}]({f})\n" for f in figures]
    lines += [
        "",
        "## Provenance",
        "",
        "```json",
        json.dumps(report["source_generations"], indent=2),
        "```",
        "",
        "Input checkpoint checksums are trusted from completed manifests; no full input checksum scan was performed.",
        "",
    ]
    temporary = child(out, "report.md.partial")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(child(out, "report.md"))


def generate(config_path, resume=False):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    root = Path(config["paths"]["work_dir"]).expanduser().resolve()
    out = root / "outputs/stats"
    if out.resolve() != root / "outputs" / "stats":
        raise ValueError("Stats output must not redirect to another directory")
    log("Reading stage manifests...")
    metadata, selected, plans, snapshot = sources(root, config)
    stages = [metadata[n]["counts"] for n in STAGES]
    log("=== Lightweight statistics ===")
    log(f"work_dir       : {root}")
    log(f"selected rows  : {metadata['select']['counts']['accepted']:,}")
    log(f"candidate rows : {metadata['score']['counts']['accepted']:,}")
    log(
        f"full scoring   : {metadata['scoring']['counts']['input']:,} (manifest only; parquet not rescanned)"
    )
    signature = digest(
        {
            "version": REPORT_VERSION,
            "script": file_hash(__file__),
            "config": config,
            "sources": snapshot,
        }
    )
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = child(out, "manifest.json")
    if resume and manifest_path.exists():
        old = read_json(manifest_path)
        if (
            old.get("complete") is True
            and old.get("signature") == signature
            and old.get("files")
            and all(
                child(out, f).is_file() and file_hash(child(out, f)) == h
                for f, h in old["files"].items()
            )
        ):
            log(f"Statistics complete. (resume: verified outputs)\nOutput: {out}")
            return old
    # An interrupted overwrite must never leave an old completion marker.
    manifest_path.unlink(missing_ok=True)
    distributions = {}
    io = {}
    for name, label in (("score", "candidate"), ("dedup", "dedup")):
        log(
            "Reading top-1M candidates..."
            if name == "score"
            else "Reading deduplicated candidates..."
        )
        paths = [(child(metadata[name]["path"], p["file"]), p) for p in plans[name]]
        io[name] = {
            "parts_opened": len(paths),
            "physical_rows": sum(p["rows"] for p in plans[name]),
            "accepted_rows": metadata[name]["counts"]["accepted"],
        }
        distributions[label] = read_distribution(
            paths, config["length_bins"], io[name]["accepted_rows"]
        )
    log("Reading selected corpus...")
    count = metadata["select"]["counts"]["accepted"]
    distributions["selected"] = read_distribution(
        [(selected, {"rows": count, "accepted": count})],
        config["length_bins"],
        count,
        selected=True,
    )
    io["selected"] = {"parts_opened": 1, "physical_rows": count, "accepted_rows": count}
    reasons = Counter()
    for s in stages:
        reasons.update(s["reject_reasons"])
    bin_rows = []
    for name, data in distributions.items():
        for i, b in enumerate(config["length_bins"]):
            actual = data["length_bins"][i]
            target = int(config["target_size"] * b["ratio"])
            bin_rows.append(
                {
                    "corpus": name,
                    "bin": bin_label(b),
                    "target_count": target,
                    "actual_count": actual,
                    "target_ratio": b["ratio"],
                    "actual_ratio": actual / max(1, data["rows"]),
                    "difference": actual - target,
                }
            )
    generations = {name: m["current"]["generation"] for name, m in metadata.items()}
    report = {
        "report_type": "lightweight",
        "source_generations": generations,
        "stages": stages,
        "reject_reasons": dict(reasons),
        "distributions": distributions,
        "target_size": config["target_size"],
        "selected_size": count,
        "shortfall": max(0, config["target_size"] - count),
        "length_bins": bin_rows,
        "io": io,
        "histogram_definitions": {
            "ja_char_count": {"width": 5, "bins": 81, "last_bin": "400+"},
            "en_word_count": {"width": 2, "bins": 101, "last_bin": "200+"},
            "bicleaner_score": {"width": 0.02, "bins": 50, "last_bin_includes": 1.0},
        },
        "omissions": {
            "full_scoring": f"全{metadata['scoring']['counts']['input']:,}件のスコア分布は、巨大なDrive再走査を避けるため再計算していない。上位候補および最終採用データの分布はexactである。",
            "early_lengths": "normalize / rule-filter等のlength histogramは省略。残存件数はmanifestのみから算出。",
        },
    }
    log("Writing figures...")
    figures = write_figures(out, stages, distributions, config["length_bins"], reasons)
    atomic_json(out, "stats.json", report)
    atomic_json(out, "effective_config.json", config)
    atomic_json(
        out,
        "summary.json",
        {
            "selected_size": count,
            "target_size": config["target_size"],
            "selected": distributions["selected"]["summary"],
            "length_bins": bin_rows,
        },
    )
    atomic_csv(
        out,
        "stages.csv",
        stages,
        ["stage", "input", "accepted", "rejected", "reject_rate", "number_of_shards", "complete"],
    )
    atomic_csv(
        out,
        "reject_reasons.csv",
        [{"reason": k, "count": v} for k, v in reasons.items()],
        ["reason", "count"],
    )
    atomic_csv(
        out,
        "length_bins.csv",
        bin_rows,
        [
            "corpus",
            "bin",
            "target_count",
            "actual_count",
            "target_ratio",
            "actual_ratio",
            "difference",
        ],
    )
    write_report(out, report, bin_rows, figures)
    files = [
        "stats.json",
        "effective_config.json",
        "summary.json",
        "stages.csv",
        "reject_reasons.csv",
        "length_bins.csv",
        "report.md",
        *figures,
    ]
    # Detect concurrent replacement of source CURRENT/exports without touching pipeline data.
    if sources(root, config)[3] != snapshot:
        raise RuntimeError(
            "Source generations changed during statistics; rerun after pipeline finishes"
        )
    manifest = {
        "complete": True,
        "signature": signature,
        "source_generations": generations,
        "source_identities": {n: m["identity"] for n, m in metadata.items()},
        "source_snapshot": snapshot,
        "files": {f: file_hash(child(out, f)) for f in files},
    }
    atomic_json(out, "manifest.json", manifest)
    log(f"Statistics complete.\nOutput: {out}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--resume", action="store_true", help="Reuse matching verified stats outputs"
    )
    args = parser.parse_args()
    generate(args.config, resume=args.resume)


if __name__ == "__main__":
    main()
