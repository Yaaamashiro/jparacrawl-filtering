import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import generate_lightweight_stats as light
import nbformat
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from conftest import ROOT, demo

from jparacrawl_filtering.pipeline import Pipeline


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def checkpoint_snapshot(root):
    return {
        str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file() and not p.is_relative_to(root / "outputs/stats")
    }


@pytest.fixture
def completed(tmp_path):
    root = tmp_path / "work"
    config = yaml.safe_load((ROOT / "configs/default.yaml").read_text(encoding="utf-8"))
    config["paths"]["work_dir"] = str(root)
    config_path = root / "config.yaml"
    root.mkdir()
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    bins = config["length_bins"]
    metadata = {}
    upstream = {"path": "original-input-not-needed"}

    def row(ja, en, score, accepted=True):
        b = next(
            i for i, b in enumerate(bins) if ja >= b["min"] and (b["max"] is None or ja <= b["max"])
        )
        return {
            "ja_char_count": ja,
            "en_word_count": en,
            "bicleaner_score": score,
            "length_bin": b,
            "accepted": accepted,
            "ja_raw": "Synthetic text only.",
        }

    candidates = [
        row(5, 3, 0.99),
        row(15, 4, 0.95),
        row(40, 9, 0.8),
        row(75, 15, 0.7),
        row(120, 30, 0.6),
    ]
    selected = [candidates[i] for i in (0, 2, 4)]
    for name in light.STAGES:
        generation = root / "data" / name / ("complete-" + name)
        generation.mkdir(parents=True)
        parts = []

        def part(filename, rows, accepted, reasons, contents=None):
            path = generation / filename
            if contents is not None:
                pq.write_table(pa.Table.from_pylist(contents), path)
            # Early stages and zero-accepted tails intentionally have NO Parquet file.
            parts.append(
                {
                    "file": filename,
                    "rows": rows,
                    "accepted": accepted,
                    "reasons": reasons,
                    "sha256": "not-rescanned",
                }
            )

        if name == "score":
            part(
                "mixed.parquet",
                6,
                5,
                {"quality_rank": 1},
                [*candidates, row(250, 100, 0.01, False)],
            )
            part("rejected-only.parquet", 16_668_655, 0, {"quality_rank": 16_668_655})
        elif name == "dedup":
            part(
                "mixed.parquet",
                5,
                4,
                {"duplicate_ja": 1, "duplicate_en": 1},
                [candidates[0], {**candidates[1], "accepted": False}, *candidates[2:]],
            )
        elif name == "select":
            part("must-not-open.parquet", 4, 3, {"length_quota": 1})
        else:
            total = 16_668_661 if name == "scoring" else 22_000_000
            part("must-not-open.parquet", total, total - 2, {"language": 2, "length": 1})
        manifest = {
            "stage": name,
            "generation": generation.name,
            "signature": "signature-" + name,
            "complete": True,
            "parts": parts,
            "upstream": upstream,
            "metrics": {},
            "settings": {"target_size": config["target_size"], "length_bins": bins},
        }
        write_json(generation / "manifest.json", manifest)
        write_json(
            generation.parent / "CURRENT.json",
            {"generation": generation.name, "signature": manifest["signature"]},
        )
        upstream = light.digest({k: manifest[k] for k in ("signature", "generation", "parts")})
        metadata[name] = manifest
    export = root / "outputs/selected/selected.parquet"
    export.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(selected), export)
    write_json(
        export.parent / "manifest.json",
        {"upstream": upstream, "rows": 3, "sha256": light.file_hash(export)},
    )
    write_json(root / "outputs/audit/manifest.json", {"complete": True})
    for name in ("accepted_sample.csv", "rejected_sample.csv"):
        (root / "outputs/audit" / name).write_text("synthetic audit", encoding="utf-8")
    (root / "checkpoints").mkdir()
    (root / "checkpoints/pipeline.lock").write_bytes(b"0")
    (root / "models").mkdir()
    (root / "models/lid.ftz").write_bytes(b"synthetic model sentinel")
    return root, config_path, config, metadata


def test_counts_and_reasons_are_manifest_only(completed, monkeypatch):
    root, _, _, _ = completed
    monkeypatch.setattr(light.pq, "ParquetFile", lambda *a, **kw: pytest.fail("Parquet opened"))
    score = light.stage_metadata(root, "score")["counts"]
    assert score["input"] == 16_668_661
    assert score["accepted"] == 5
    assert score["rejected"] == 16_668_656
    assert score["reject_reasons"] == {"quality_rank": 16_668_656}
    assert score["number_of_shards"] == 2 and score["complete"]
    dedup = light.stage_metadata(root, "dedup")["counts"]
    assert dedup["rejected"] == 1
    assert sum(dedup["reject_reasons"].values()) == 2


def test_generate_reads_only_bounded_projected_accepted_parts(completed, monkeypatch, capsys):
    root, config_path, _, _ = completed
    before = checkpoint_snapshot(root)
    src_before = {p: p.read_bytes() for p in (ROOT / "src/jparacrawl_filtering").glob("*.py")}
    opened, columns = [], []
    real_parquet = light.pq.ParquetFile

    class GuardedParquet:
        def __init__(self, path):
            path = Path(path)
            assert path.name in ("mixed.parquet", "selected.parquet")
            assert path.parent.parent.name in ("score", "dedup", "outputs")
            opened.append(path)
            self.reader = real_parquet(path)
            self.metadata = self.reader.metadata

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.reader.close()

        def iter_batches(self, **kwargs):
            columns.append(kwargs["columns"])
            assert "ja_raw" not in kwargs["columns"]
            return self.reader.iter_batches(**kwargs)

    monkeypatch.setattr(light.pq, "ParquetFile", GuardedParquet)
    manifest = light.generate(config_path)
    assert len(opened) == 3
    assert columns == [list(light.COLUMNS) + ["accepted"]] * 2 + [list(light.COLUMNS)]
    stats = light.read_json(root / "outputs/stats/stats.json")
    assert stats["distributions"]["candidate"]["length_bins"] == [1, 1, 1, 1, 1]
    assert stats["distributions"]["dedup"]["length_bins"] == [1, 0, 1, 1, 1]
    selected = stats["distributions"]["selected"]
    assert selected["length_bins"] == [1, 0, 1, 0, 1]
    assert selected["summary"]["ja_char_count"]["median"] == 40
    assert selected["summary"]["en_word_count"]["mean"] == 14
    assert selected["summary"]["bicleaner_score"]["p10"] == pytest.approx(0.64)
    assert sum(selected["histograms"]["bicleaner_score"]) == 3
    quota = [b for b in stats["length_bins"] if b["corpus"] == "selected"]
    assert [b["target_count"] for b in quota] == [6000, 45000, 120000, 99000, 30000]
    assert quota[0]["difference"] == 1 - 6000
    assert stats["io"]["score"]["physical_rows"] == 6
    assert "16,668,661" in stats["omissions"]["full_scoring"]
    assert manifest["complete"] is True
    for name, checksum in manifest["files"].items():
        assert light.file_hash(root / "outputs/stats" / name) == checksum
    for name in (
        "stage_retention",
        "selected_ja_length",
        "selected_en_words",
        "selected_bicleaner_scores",
        "length_bin_comparison",
        "reject_reasons",
    ):
        assert (root / "outputs/stats" / (name + ".png")).read_bytes().startswith(b"\x89PNG")
    assert before == checkpoint_snapshot(root)
    assert all(p.read_bytes() == content for p, content in src_before.items())
    assert "Statistics complete." in capsys.readouterr().out


def test_manifest_is_last_and_failed_overwrite_is_incomplete(completed, monkeypatch):
    root, config_path, _, _ = completed
    before = checkpoint_snapshot(root)
    out = root / "outputs/stats"
    writes = []
    real_json = light.atomic_json

    def tracked(out, name, value):
        assert not (out / "manifest.json").exists()
        writes.append(name)
        real_json(out, name, value)

    monkeypatch.setattr(light, "atomic_json", tracked)
    light.generate(config_path)
    assert writes[-1] == "manifest.json"

    def interrupted(*args):
        assert not (out / "manifest.json").exists()
        raise RuntimeError("runtime disconnected")

    monkeypatch.setattr(light, "write_figures", interrupted)
    with pytest.raises(RuntimeError, match="disconnected"):
        light.generate(config_path)
    assert not (out / "manifest.json").exists()
    assert before == checkpoint_snapshot(root)


def test_resume_checks_outputs_without_reading_parquet(completed, monkeypatch):
    root, config_path, _, _ = completed
    light.generate(config_path)
    monkeypatch.setattr(light.pq, "ParquetFile", lambda *a: pytest.fail("Resume opened Parquet"))
    light.generate(config_path, resume=True)
    (root / "outputs/stats/stats.json").write_text("corrupt", encoding="utf-8")
    with pytest.raises(pytest.fail.Exception, match="Resume opened"):
        light.generate(config_path, resume=True)
    assert not (root / "outputs/stats/manifest.json").exists()


@pytest.mark.parametrize("case", ["incomplete", "mixed", "export", "quota", "too_many_rows"])
def test_invalid_sources_fail_before_parquet_scan(completed, monkeypatch, case):
    root, config_path, config, _ = completed
    if case == "quota":
        config["target_size"] = 42
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    elif case == "export":
        path = root / "outputs/selected/manifest.json"
        value = light.read_json(path)
        value["upstream"] = "another-run"
        write_json(path, value)
    else:
        stage = "score"
        path = root / "data" / stage / ("complete-" + stage) / "manifest.json"
        value = light.read_json(path)
        if case == "incomplete":
            value["complete"] = False
        elif case == "mixed":
            value["upstream"] = "another-generation"
        else:
            # Change tail metadata without breaking the downstream identity chain.
            # Check budget independently in sources below, using patched stage_metadata.
            original = light.stage_metadata

            def inflated(root, name):
                result = original(root, name)
                if name == "score":
                    result["manifest"]["parts"][0]["rows"] = light.MAX_SCAN_ROWS + 1
                return result

            monkeypatch.setattr(light, "stage_metadata", inflated)
        write_json(path, value)
    before = checkpoint_snapshot(root)
    monkeypatch.setattr(light.pq, "ParquetFile", lambda *a: pytest.fail("Unsafe scan"))
    with pytest.raises(ValueError, match="bounded scan limit" if case == "too_many_rows" else None):
        light.generate(config_path)
    assert before == checkpoint_snapshot(root)


def test_concurrent_generation_change_prevents_completion(completed, monkeypatch):
    root, config_path, _, _ = completed
    real_sources = light.sources
    calls = 0

    def changed(*args):
        nonlocal calls
        calls += 1
        result = real_sources(*args)
        if calls == 2:
            result[3]["selected_file"]["size"] += 1
        return result

    monkeypatch.setattr(light, "sources", changed)
    with pytest.raises(RuntimeError, match="changed"):
        light.generate(config_path)
    assert not (root / "outputs/stats/manifest.json").exists()


def test_cli_does_not_import_pipeline_package(completed):
    root, config_path, _, _ = completed
    script = ROOT / "scripts/generate_lightweight_stats.py"
    program = """
import runpy, sys
sys.argv = [sys.argv[1], '--config', sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
assert not any(n == 'jparacrawl_filtering' or n.startswith('jparacrawl_filtering.') for n in sys.modules)
"""
    env = {**os.environ, "MPLBACKEND": "module://matplotlib_inline.backend_inline"}
    result = subprocess.run(
        [sys.executable, "-c", program, str(script), str(config_path)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert light.read_json(root / "outputs/stats/manifest.json")["complete"]


def test_actual_pipeline_checkpoint_format_is_compatible(prepared):
    stores = Pipeline(
        prepared, lid=demo.SyntheticLanguageID(), scorer=demo.SyntheticScorer()
    ).execute(until="select")
    root = Path(prepared["paths"]["work_dir"])
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(prepared), encoding="utf-8")
    before = checkpoint_snapshot(root)
    manifest = light.generate(config_path)
    assert manifest["source_identities"] == {name: store.identity for name, store in stores.items()}
    report = light.read_json(root / "outputs/stats/stats.json")
    assert report["selected_size"] == 4
    assert report["distributions"]["candidate"]["rows"] == stores["score"].counts()["accepted"]
    assert before == checkpoint_snapshot(root)


def test_notebook_stats_bypasses_pipeline():
    nb = nbformat.read(ROOT / "notebooks/jparacrawl_filtering_colab.ipynb", as_version=4)
    cell = next(
        c.source
        for c in nb.cells
        if c.cell_type == "code" and "generate_lightweight_stats.py" in c.source
    )
    assert "PIPELINE_PYTHON" in cell and "ops.stream_command" in cell
    assert "WORKER_ENV" in cell and "ops.validate_run" in cell
    assert "stage_retention.png" in cell
    for c in nb.cells:
        if c.cell_type == "code":
            for node in ast.walk(ast.parse(c.source)):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert not (
                        node.func.id == "run_stage"
                        and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value == "stats"
                    )
