import csv
import gzip
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from conftest import ROOT, demo

from jparacrawl_filtering.cli import main
from jparacrawl_filtering.pipeline import Pipeline
from jparacrawl_filtering.storage import StageStore, input_rows, pipeline_lock


def run(c, **kw):
    return Pipeline(
        c, lid=demo.SyntheticLanguageID(), scorer=demo.SyntheticScorer(), **kw
    ).execute()


def test_end_to_end_and_resume(prepared):
    stores = run(prepared)
    out = Path(prepared["paths"]["work_dir"]) / "outputs"
    assert stores["select"].counts()["accepted"] == 4
    final = pq.read_table(out / "selected/selected.parquet").to_pylist()
    assert len(final) == 4 and all(r["accepted"] for r in final)
    assert len({r["ja_norm"] for r in final}) == len(final)
    assert len({r["en_norm"] for r in final}) == len(final)
    assert all(r["estimated_duration"] is None and r["actual_duration"] is None for r in final)
    stats = json.loads((out / "stats/stats.json").read_text(encoding="utf-8"))
    assert stats["stages"][0]["input"] == 16
    assert sum(s["rejected"] for s in stats["stages"]) + 4 == 16
    assert stats["reject_reasons"]["not_sentence_like"] >= 1
    assert len(list((out / "stats").glob("*.png"))) == 5
    with (out / "audit/rejected_sample.csv").open(encoding="utf-8-sig", newline="") as f:
        audit = list(csv.DictReader(f))
    assert len(audit) == 12 and len({r["pair_id"] for r in audit}) == 12
    identities = {n: s.identity for n, s in stores.items()}
    mtime = (out / "audit/accepted_sample.csv").stat().st_mtime_ns
    resumed = run(prepared, resume=True)
    assert identities == {n: s.identity for n, s in resumed.items()}
    assert mtime == (out / "audit/accepted_sample.csv").stat().st_mtime_ns
    with pytest.raises(FileExistsError):
        run(prepared)


def test_config_changes_and_force_invalidate_descendants(prepared):
    before = run(prepared)
    changed = deepcopy(prepared)
    changed["target_size"] = 3
    after = run(changed, resume=True)
    assert before["scoring"].identity == after["scoring"].identity
    assert before["dedup"].identity == after["dedup"].identity
    assert before["select"].identity != after["select"].identity
    forced = run(changed, resume=True, force=["speech-filter"])
    assert forced["rule-filter"].identity == after["rule-filter"].identity
    assert forced["scoring"].identity != after["scoring"].identity
    assert forced["select"].identity != after["select"].identity


def test_candidate_limit_changes_reuse_inference(prepared):
    before = run(prepared)
    prepared["bicleaner"]["candidate_size"] = 6
    after = run(prepared, resume=True)
    assert before["scoring"].identity == after["scoring"].identity
    assert before["score"].identity != after["score"].identity
    assert after["score"].counts()["accepted"] == 6
    assert after["score"].counts()["reject_reasons"]["quality_rank"] == 1


def test_partial_scoring_resume(prepared):
    class FailingScorer(demo.SyntheticScorer):
        calls = 0
        fail = True

        def score(self, rows):
            self.calls += 1
            if self.fail and self.calls == 2:
                raise RuntimeError("Synthetic interruption")
            return super().score(rows)

    scorer = FailingScorer()
    p = Pipeline(prepared, lid=demo.SyntheticLanguageID(), scorer=scorer)
    with pytest.raises(RuntimeError, match="Synthetic interruption"):
        p.execute(until="score")
    assert len(p.stores["scoring"].manifest["parts"]) == 1
    first = p.stores["scoring"].path / "part-000000.parquet"
    mtime = first.stat().st_mtime_ns
    scorer.fail = False
    Pipeline(prepared, resume=True, lid=demo.SyntheticLanguageID(), scorer=scorer).execute(
        until="score"
    )
    assert first.stat().st_mtime_ns == mtime


def test_checkpoint_corruption_fails_safely(tmp_path):
    s = StageStore(tmp_path, "test", {}, "raw")
    s.write(0, [{"accepted": True, "reject_reason": ""}])
    s.finish()
    (s.path / "part-000000.parquet").write_bytes(b"bad")
    with pytest.raises(ValueError, match="Corrupt checkpoint"):
        StageStore(tmp_path, "test", {}, "raw", resume=True)
    assert not StageStore(tmp_path, "test", {}, "raw", force=True).complete


def test_empty_pipeline_exports_valid_empty_parquet(prepared):
    Path(prepared["paths"]["input"]).write_text("", encoding="utf-8")
    stores = run(prepared)
    assert stores["select"].counts()["accepted"] == 0
    out = Path(prepared["paths"]["work_dir"]) / "outputs"
    assert pq.read_table(out / "selected/selected.parquet").num_rows == 0
    stats = json.loads((out / "stats/stats.json").read_text(encoding="utf-8"))
    assert stats["shortfall"] == prepared["target_size"]


def test_gzip_mapping_and_malformed_input(config, tmp_path):
    p = tmp_path / "input.tsv.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write('ja\ten\n日本語です。\t"This is Japanese."\n')
    config["paths"]["input"] = str(p)
    config["input"].update(header=True, columns={"ja": "ja", "en": "en"}, expected_columns=2)
    rows = list(input_rows(config))
    assert rows[0]["en_raw"] == '"This is Japanese."'
    assert rows[0]["source_line"] == 2
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.write("ja\ten\nbroken\n")
    with pytest.raises(ValueError, match="column count"):
        list(input_rows(config))


def test_cpu_import_and_cli():
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import jparacrawl_filtering.cli,sys; assert 'tensorflow' not in sys.modules; assert 'torch' not in sys.modules; assert 'fasttext' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert main(["validate-config", "--config", str(ROOT / "configs/default.yaml")]) == 0
    assert main(["run", "--target-size", "-1"]) == 1
    assert (
        subprocess.run(
            [sys.executable, "-m", "jparacrawl_filtering", "--help"], capture_output=True
        ).returncode
        == 0
    )


def test_lock_prevents_two_writers(tmp_path):
    with pipeline_lock(tmp_path):
        with pytest.raises(RuntimeError, match="locked"):
            with pipeline_lock(tmp_path):
                pass
