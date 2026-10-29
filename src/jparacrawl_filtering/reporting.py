"""Seeded reservoir audit and exact streaming histograms with bounded memory."""

import csv
import random
from collections import Counter
from pathlib import Path

from .pipeline import DATA_STAGES
from .storage import atomic_json, file_hash, read_json


def export_selected(pipeline):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from .schema import SCHEMA
    from .storage import chunks

    out = pipeline.root / "outputs" / "selected"
    out.mkdir(parents=True, exist_ok=True)
    meta = out / "manifest.json"
    p = out / "selected.parquet"
    identity = pipeline.stores["select"].identity
    if meta.exists() and p.exists():
        old = read_json(meta)
        if old["upstream"] == identity and old["sha256"] == file_hash(p):
            return
    temp = p.with_suffix(".partial")
    with pq.ParquetWriter(temp, SCHEMA, compression="zstd") as writer:
        for batch in chunks(pipeline.stores["select"].rows(True), pipeline.c["batch_size"]):
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
    temp.replace(p)
    atomic_json(
        meta,
        {"upstream": identity, "sha256": file_hash(p), "rows": pq.ParquetFile(p).metadata.num_rows},
    )


def rejected_rows(pipeline):
    for name in DATA_STAGES:
        for row in pipeline.stores[name].rows():
            if not row["accepted"]:
                yield row


def reservoir(rows, size, seed):
    rng = random.Random(seed)
    sample = []
    for i, row in enumerate(rows):
        if len(sample) < size:
            sample.append(row)
        elif size:
            j = rng.randrange(i + 1)
            if j < size:
                sample[j] = row
    return sample


def csv_write(path, rows, fields):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".partial")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(p)


def export_state(pipeline, name, settings):
    from .storage import digest

    out = pipeline.root / "outputs" / name
    out.mkdir(parents=True, exist_ok=True)
    signature = digest(
        {"settings": settings, "upstream": {n: pipeline.stores[n].identity for n in DATA_STAGES}}
    )
    manifest = out / "manifest.json"
    if manifest.exists() and name not in pipeline.force:
        old = read_json(manifest)
        if old["signature"] == signature and pipeline.resume:
            if all(
                (out / f).is_file() and file_hash(out / f) == h for f, h in old["files"].items()
            ):
                return out, signature, True
    return out, signature, False


def finish_export(out, signature, files):
    atomic_json(
        out / "manifest.json",
        {"signature": signature, "files": {f: file_hash(out / f) for f in files}},
    )


def audit(pipeline):
    c = pipeline.c
    out, sig, skip = export_state(
        pipeline,
        "audit",
        {"seed": c["seed"], "audit": c["audit"], "training_text": c["training_text"]},
    )
    if skip:
        return
    accepted = reservoir(
        pipeline.stores["select"].rows(True), c["audit"]["accepted_samples"], c["seed"]
    )
    rejected = reservoir(rejected_rows(pipeline), c["audit"]["rejected_samples"], c["seed"] + 1)
    suffix = "raw" if c["training_text"] == "raw" else "norm"
    for row in accepted + rejected:
        row["ja"], row["en"] = row[f"ja_{suffix}"], row[f"en_{suffix}"]
    fields = [
        "pair_id",
        "ja",
        "en",
        "bicleaner_score",
        "length_bin",
        "ja_char_count",
        "en_word_count",
        "accepted",
        "reject_reason",
        "stage",
        "first_reject_stage",
    ]
    csv_write(out / "accepted_sample.csv", accepted, fields)
    csv_write(out / "rejected_sample.csv", rejected, fields)
    finish_export(out, sig, ["accepted_sample.csv", "rejected_sample.csv"])


def histogram(rows, key, width, bins, accepted_only=False):
    counts = [0] * bins
    for row in rows:
        value = row.get(key)
        if value is not None and (not accepted_only or row["accepted"]):
            counts[min(bins - 1, max(0, int(value / width)))] += 1
    return counts


def statistics(pipeline):
    out, sig, skip = export_state(pipeline, "stats", {"report_version": 1, "config": pipeline.c})
    if skip:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = [pipeline.stores[n].counts() for n in DATA_STAGES]
    reasons = Counter()
    for s in stages:
        reasons.update(s["reject_reasons"])
    selected_bins = Counter()
    for row in pipeline.stores["select"].rows(True):
        selected_bins[row["length_bin"]] += 1
    report = {
        "stages": stages,
        "reject_reasons": dict(reasons),
        "selected_length_bins": dict(selected_bins),
        "target_size": pipeline.c["target_size"],
        "selected_size": stages[-1]["accepted"],
        "shortfall": max(0, pipeline.c["target_size"] - stages[-1]["accepted"]),
        "inference": pipeline.stores["scoring"].counts(),
        "bin_definitions": pipeline.c["length_bins"],
    }
    atomic_json(out / "stats.json", report)
    atomic_json(out / "effective_config.json", pipeline.c)
    fields = ["stage", "input", "accepted", "rejected", "reject_rate"]
    csv_write(out / "stages.csv", stages, fields)
    csv_write(
        out / "reject_reasons.csv",
        [{"reason": k, "count": v} for k, v in reasons.items()],
        ["reason", "count"],
    )
    files = ["stats.json", "effective_config.json", "stages.csv", "reject_reasons.csv"]
    hists = {}
    for key, width, count, xlabel, filename in (
        ("ja_char_count", 5, 81, "Japanese characters (last bin includes overflow)", "ja_length"),
        ("en_word_count", 2, 101, "English words (last bin includes overflow)", "en_words"),
    ):
        fig, ax = plt.subplots(figsize=(7, 4))
        values = {}
        for name, label in (
            ("normalize", "Before rule filtering"),
            ("rule-filter", "After rules"),
            ("select", "Selected"),
        ):
            values[name] = histogram(pipeline.stores[name].rows(), key, width, count, True)
            total = sum(values[name])
            ax.stairs(
                [n / max(1, total) for n in values[name]],
                [i * width for i in range(count + 1)],
                label=label,
            )
        ax.set(xlabel=xlabel, ylabel="Fraction of pairs")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{filename}.png", dpi=180)
        plt.close(fig)
        hists[filename] = values
        files.append(f"{filename}.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    scores = {}
    for name, label in (("scoring", "All scored"), ("select", "Selected")):
        scores[name] = histogram(pipeline.stores[name].rows(), "bicleaner_score", 0.02, 50, True)
        ax.stairs(scores[name], [i * 0.02 for i in range(51)], label=label)
    ax.set(xlabel="Bicleaner-AI probability", ylabel="Pairs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "bicleaner_scores.png", dpi=180)
    plt.close(fig)
    files.append("bicleaner_scores.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    selected = histogram(pipeline.stores["select"].rows(True), "ja_char_count", 5, 81)
    rejected = histogram(rejected_rows(pipeline), "ja_char_count", 5, 81)
    for values, label in ((selected, "Selected"), (rejected, "Rejected at any stage")):
        ax.stairs(
            [n / max(1, sum(values)) for n in values], [i * 5 for i in range(82)], label=label
        )
    ax.set(xlabel="Japanese characters (last bin includes overflow)", ylabel="Fraction of pairs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "selected_vs_rejected.png", dpi=180)
    plt.close(fig)
    files.append("selected_vs_rejected.png")
    fig, ax = plt.subplots(figsize=(8, max(3, len(reasons) * 0.26)))
    items = sorted(reasons.items(), key=lambda kv: kv[1])
    ax.barh([k for k, _ in items], [v for _, v in items])
    ax.set(xlabel="Reason occurrences (a pair may have several)")
    fig.tight_layout()
    fig.savefig(out / "reject_reasons.png", dpi=180)
    plt.close(fig)
    files.append("reject_reasons.png")
    atomic_json(
        out / "histograms.json",
        {**hists, "scores": scores, "selected_length": selected, "rejected_length": rejected},
    )
    files.append("histograms.json")
    lines = [
        "# Filtering report",
        "",
        f"Selected **{report['selected_size']:,}** / target **{report['target_size']:,}** pairs. "
        f"Shortfall: **{report['shortfall']:,}**.",
        "",
        "| Stage | Input | Accepted | Rejected | Reject rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {s['stage']} | {s['input']:,} | {s['accepted']:,} | {s['rejected']:,} | {s['reject_rate']:.2%} |"
        for s in stages
    )
    lines.extend(
        [
            "",
            "## Length bins",
            "",
            "| Japanese characters | Target ratio | Selected | Actual ratio |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for i, b in enumerate(pipeline.c["length_bins"]):
        label = f"{b['min']}–{b['max']}" if b["max"] is not None else f"{b['min']}+"
        count = selected_bins[i]
        lines.append(
            f"| {label} | {b['ratio']:.1%} | {count:,} | {count / max(1, report['selected_size']):.1%} |"
        )
    lines.extend(
        [
            "",
            "## Reject reasons",
            "",
            "A pair may have multiple reasons.",
            "",
            "| Reason | Occurrences |",
            "| --- | ---: |",
        ]
    )
    lines.extend(f"| {reason} | {count:,} |" for reason, count in reasons.most_common())
    lines.extend(["", "## Figures", ""])
    lines.extend(f"![{f.removesuffix('.png')}]({f})\n" for f in files if f.endswith(".png"))
    lines.extend(
        [
            "",
            "Inspect the audit CSV before TTS; ranking scores are not a quality guarantee.",
            "",
            "See [stats.json](stats.json) for inference metrics and [effective config](effective_config.json).",
            "",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    files.append("report.md")
    finish_export(out, sig, files)
