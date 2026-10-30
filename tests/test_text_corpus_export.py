import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import export_text_corpus as corpus
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from conftest import ROOT, demo

from jparacrawl_filtering.pipeline import Pipeline


@pytest.fixture(autouse=True)
def local_scratch(tmp_path, monkeypatch):
    # The Windows sandbox's redirected system temp does not support SQLite.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    for name in ("TMPDIR", "TMP", "TEMP"):
        monkeypatch.setenv(name, str(tmp_path))


def snapshot(root):
    return {
        str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file() and not p.is_relative_to(root / "outputs/text-corpus")
    }


def make_source(tmp_path, rows=None, training_text="normalized"):
    root = tmp_path / "work"
    source = root / "outputs/selected/selected.parquet"
    source.parent.mkdir(parents=True)
    if rows is None:
        bins = [0] * 2 + [1] * 3 + [2] * 6 + [3] * 7 + [4] * 2
        rows = []
        for index, b in enumerate(bins):
            rows.append(
                {
                    "pair_id": f"original-pair-{index * 19 + 103}",
                    "ja_raw": f" 自作のテスト文　{index}。 ",
                    "en_raw": f" Synthetic  text {index}. ",
                    "ja_norm": f"自作のテスト文 {index}。",
                    "en_norm": f"Synthetic text {index}.",
                    "bicleaner_score": 0.8 + index / 100,
                    "length_bin": b,
                    "source_line": index * 19 + 103,
                    "accepted": True,
                }
            )
    suffix = "raw" if training_text == "raw" else "norm"
    rows = [{**r, "ja": r["ja_" + suffix], "en": r["en_" + suffix]} for r in rows]
    pq.write_table(pa.Table.from_pylist(rows), source)
    config = {"paths": {"work_dir": str(root)}, "training_text": training_text}
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    (root / "outputs/audit").mkdir()
    (root / "outputs/audit/accepted_sample.csv").write_text("synthetic audit", encoding="utf-8")
    return root, config_path, source, rows


def read_records(root):
    return [
        json.loads(line)
        for line in (root / "outputs/text-corpus/manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]


def test_exact_production_allocation():
    counts = {0: 6000, 1: 45000, 2: 120000, 3: 99000, 4: 30000}
    dev = corpus.allocate_dev(counts, 2000)
    assert dev == {0: 40, 1: 300, 2: 800, 3: 660, 4: 200}
    assert sum(dev.values()) == 2000
    assert sum(counts[b] - dev[b] for b in counts) == 298000


def test_quota_rounding_is_stratified_and_ties_are_stable():
    counts = {4: 2, 3: 7, 2: 6, 1: 3, 0: 2}
    quotas = corpus.allocate_dev(counts, 4)
    assert quotas == {0: 1, 1: 1, 2: 1, 3: 1, 4: 0}
    assert sum(quotas.values()) == 4
    for b in counts:
        assert abs(quotas[b] - counts[b] * 4 / 20) < 1
    assert corpus.allocate_dev({0: 1, 1: 1}, 0) == {0: 0, 1: 0}
    assert corpus.allocate_dev({0: 1, 1: 1}, 2) == {0: 1, 1: 1}


def test_deterministic_export_hash_ids_schema_and_no_leakage(tmp_path):
    root, config, source, rows = make_source(tmp_path)
    before = snapshot(root)
    first = corpus.export(config, expected_total=20, dev_size=4)
    manifest = root / "outputs/text-corpus/manifest.jsonl"
    first_bytes = manifest.read_bytes()
    records = read_records(root)
    assert [r["pair_id"] for r in records] == [r["pair_id"] for r in rows]
    assert len({r["pair_id"] for r in records}) == 20
    assert Counter(r["split"] for r in records) == {"train": 16, "dev": 4}
    for r, original in zip(records, rows):
        assert set(r) == {
            "pair_id",
            "split",
            "ja_text",
            "en_text",
            "corpus",
            "bicleaner_score",
            "length_bin",
            "source_line",
        }
        assert r["ja_text"] == original["ja_norm"]
        assert r["en_text"] == original["en_norm"]
        assert r["corpus"] == "jparacrawl_v3_filtered"
        assert r["bicleaner_score"] == original["bicleaner_score"]
        assert r["source_line"] == original["source_line"]
    train_ids = {r["pair_id"] for r in records if r["split"] == "train"}
    dev_ids = {r["pair_id"] for r in records if r["split"] == "dev"}
    assert not train_ids & dev_ids
    train_text = {(r["ja_text"], r["en_text"]) for r in records if r["split"] == "train"}
    dev_text = {(r["ja_text"], r["en_text"]) for r in records if r["split"] == "dev"}
    assert not train_text & dev_text
    assert first["text_fields"] == {"ja": "ja_norm", "en": "en_norm"}
    assert first["source_sha256"] == corpus.file_hash(source)
    assert first["output_sha256"] == corpus.file_hash(manifest)
    assert first["length_bin_counts"]["dev"] == {"0": 1, "1": 1, "2": 1, "3": 1, "4": 0}
    assert first["test_pairs"] == 0
    assert len(list((root / "outputs/text-corpus").iterdir())) == 2
    second = corpus.export(config, expected_total=20, dev_size=4)
    assert first_bytes == manifest.read_bytes()
    assert first == second
    assert before == snapshot(root)


def test_raw_config_is_respected_without_normalizing(tmp_path):
    root, config, _, rows = make_source(tmp_path, training_text="raw")
    summary = corpus.export(config, expected_total=20, dev_size=4)
    assert summary["text_fields"] == {"ja": "ja_raw", "en": "en_raw"}
    for original, result in zip(rows, read_records(root)):
        assert result["ja_text"] == original["ja_raw"]
        assert result["en_text"] == original["en_raw"]


def test_different_seed_changes_dev_membership(tmp_path):
    root, config, _, _ = make_source(tmp_path)
    corpus.export(config, expected_total=20, dev_size=4, seed=42)
    first_dev = {r["pair_id"] for r in read_records(root) if r["split"] == "dev"}
    corpus.export(config, expected_total=20, dev_size=4, seed=43)
    second_dev = {r["pair_id"] for r in read_records(root) if r["split"] == "dev"}
    assert first_dev != second_dev


@pytest.mark.parametrize(
    "problem", ["duplicate_id", "empty_ja", "empty_en", "aliases", "leakage", "bad_bin"]
)
def test_invalid_corpus_is_rejected_without_touching_source(tmp_path, problem):
    root, config, source, rows = make_source(tmp_path)
    if problem == "duplicate_id":
        rows[1]["pair_id"] = rows[0]["pair_id"]
    elif problem == "empty_ja":
        rows[0]["ja_norm"] = " \t\n"
    elif problem == "empty_en":
        rows[0]["en_norm"] = ""
    elif problem == "aliases":
        rows[0]["ja"] = "Disagrees with selected training text."
    elif problem == "leakage":
        # Put all rows into one bin with identical text: any nonempty dev will leak.
        for r in rows:
            r.update(
                length_bin=0,
                ja_norm="同じ自作文です。",
                en_norm="Same synthetic sentence.",
                ja="同じ自作文です。",
                en="Same synthetic sentence.",
            )
    else:
        rows[0]["length_bin"] = None
    pq.write_table(pa.Table.from_pylist(rows), source)
    before = snapshot(root)
    with pytest.raises(ValueError):
        corpus.export(config, expected_total=20, dev_size=4)
    assert before == snapshot(root)
    assert not (root / "outputs/text-corpus/summary.json").exists()
    assert not (root / "outputs/text-corpus/manifest.jsonl").exists()


def test_wrong_total_uses_production_guard_and_preserves_previous_export(tmp_path):
    root, config, _, _ = make_source(tmp_path)
    corpus.export(config, expected_total=20, dev_size=4)
    out = root / "outputs/text-corpus"
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(ValueError, match="300,000"):
        corpus.export(config)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_selected_manifest_checksum_is_verified(tmp_path):
    root, config, source, _ = make_source(tmp_path)
    (source.parent / "manifest.json").write_text(
        json.dumps({"rows": 20, "sha256": "wrong"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="SHA256"):
        corpus.export(config, expected_total=20, dev_size=4)
    assert not (root / "outputs/text-corpus/manifest.jsonl").exists()


def test_checkpoint_config_mismatch_and_real_schema(prepared):
    root = Path(prepared["paths"]["work_dir"])
    stores = Pipeline(
        prepared, lid=demo.SyntheticLanguageID(), scorer=demo.SyntheticScorer()
    ).execute(until="select")
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(prepared), encoding="utf-8")
    before = snapshot(root)
    summary = corpus.export(config_path, expected_total=4, dev_size=1)
    assert summary["provenance"]["select_identity"] == stores["select"].identity
    assert summary["provenance"]["training_aliases_checked"]
    assert {f["name"] for f in summary["source_schema"]} >= {
        "pair_id",
        "ja_raw",
        "en_raw",
        "ja_norm",
        "en_norm",
        "bicleaner_score",
        "length_bin",
        "source_line",
    }
    assert before == snapshot(root)
    prepared["training_text"] = "raw"
    config_path.write_text(yaml.safe_dump(prepared), encoding="utf-8")
    with pytest.raises(ValueError, match="training_text differs"):
        corpus.export(config_path, expected_total=4, dev_size=1)


def test_cli_uses_no_pipeline_import_and_optional_metadata_is_optional(tmp_path):
    root, config, source, rows = make_source(tmp_path)
    rows = [
        {k: v for k, v in row.items() if k not in ("bicleaner_score", "source_line")}
        for row in rows
    ]
    pq.write_table(pa.Table.from_pylist(rows), source)
    script = ROOT / "scripts/export_text_corpus.py"
    program = """
import runpy, sys
sys.argv = [sys.argv[1], '--config', sys.argv[2], '--expected-total', '20', '--dev-size', '4']
runpy.run_path(sys.argv[0], run_name='__main__')
assert not any(n == 'jparacrawl_filtering' or n.startswith('jparacrawl_filtering.') for n in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(script), str(config)],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr
    assert all("source_line" not in r and "bicleaner_score" not in r for r in read_records(root))
