"""Stage orchestration and parameter-aware resume without loading entire corpora."""

import logging
import time
from importlib import metadata
from pathlib import Path

from .config import validate
from .filters import FastTextLanguageID, normalize_row, rule_filter, speech_filter
from .ranking import deduplicate, exact_deduplicate, rank_candidates, select_rows
from .scoring import BicleanerScorer
from .storage import (
    StageStore,
    chunks,
    digest,
    file_hash,
    input_rows,
    pipeline_lock,
    source_identity,
)

LOG = logging.getLogger(__name__)
DATA_STAGES = (
    "normalize",
    "rule-filter",
    "speech-filter",
    "exact-dedup",
    "score",
    "dedup",
    "select",
)
STAGES = (*DATA_STAGES, "audit", "stats")
SETTINGS = {
    "normalize": ("input",),
    "rule-filter": ("language_id", "length", "numbers", "symbols", "navigation", "boilerplate"),
    "speech-filter": ("sentence",),
    "exact-dedup": (),
    "scoring": ("bicleaner",),
    "score": ("bicleaner",),
    "dedup": ("one_side_dedup",),
    "select": ("target_size", "length_bins", "training_text"),
    "audit": ("seed", "audit", "training_text"),
    "stats": (),
}


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


class Pipeline:
    def __init__(self, config, resume=False, force=(), lid=None, scorer=None):
        self.c = validate(config)
        self.root = Path(config["paths"]["work_dir"]).resolve()
        self.resume = resume
        self.force = set(force)
        unknown = self.force - set(STAGES)
        if unknown:
            raise ValueError(f"Unknown forced stages: {sorted(unknown)}")
        self.lid = lid
        self.scorer = scorer
        self.injected_lid = type(lid).__name__ if lid is not None else None
        self.injected_scorer = type(scorer).__name__ if scorer is not None else None
        self.code_digest = digest(
            {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        )
        self.stores = {}
        self.scratch = (
            Path(config["paths"].get("scratch_dir", self.root / ".work"))
            / digest(str(self.root))[:16]
        )

    def store(self, name, upstream):
        settings = {key: self.c[key] for key in SETTINGS[name]}
        settings["batch_size"] = self.c["batch_size"]
        settings["code_digest"] = self.code_digest
        if name == "rule-filter":
            if self.injected_lid is None:
                model = Path(self.c["language_id"]["model_path"])
                settings["lid_model_sha256"] = file_hash(model)
                settings["fasttext_version"] = package_version("fasttext-wheel")
            else:
                settings["test_lid"] = self.injected_lid
        if name in ("scoring", "score"):
            settings["bicleaner_version"] = package_version("bicleaner-ai")
            settings["injected_scorer"] = self.injected_scorer
            if name == "scoring":
                settings["bicleaner"] = {
                    k: v for k, v in self.c["bicleaner"].items() if k != "candidate_size"
                }
            model_path = Path(self.c["bicleaner"]["model"])
            if model_path.is_dir():
                settings["local_model_files"] = [
                    (str(p.relative_to(model_path)), p.stat().st_size, p.stat().st_mtime_ns)
                    for p in sorted(model_path.rglob("*"))
                    if p.is_file()
                ]
        force = name in self.force or (name == "scoring" and "score" in self.force)
        store = StageStore(self.root, name, settings, upstream, self.resume, force)
        self.stores[name] = store
        return store

    def transform(self, store, rows, function, batch_size=None):
        if store.complete:
            LOG.info("stage=%s resume=skip checkpoint=%s", store.manifest["stage"], store.path)
            return store
        started = time.monotonic()
        for i, batch in enumerate(chunks(rows, batch_size or self.c["batch_size"])):
            if i < len(store.manifest["parts"]):
                continue
            store.write(i, function(batch))
        store.finish({"elapsed_this_run_seconds": time.monotonic() - started})
        return store

    def execute(self, command="run", until=None):
        if command not in ("run", *STAGES):
            raise ValueError(f"Unknown stage: {command}")
        stop = until or ("stats" if command == "run" else command)
        if stop not in STAGES:
            raise ValueError("Unknown --until stage")
        # Stage commands check/run prerequisite stages, so stale upstream config can
        # never silently feed a new downstream run. --resume skips valid prerequisites.
        with pipeline_lock(self.root):
            self._execute(stop)
        return self.stores

    def _execute(self, stop):
        c = self.c
        norm = self.store("normalize", source_identity(c))
        self.transform(norm, input_rows(c), lambda batch: [normalize_row(r) for r in batch])
        if stop == "normalize":
            return
        rules = self.store("rule-filter", norm.identity)
        if not rules.complete:
            self.lid = self.lid or FastTextLanguageID(c["language_id"]["model_path"])
        self.transform(
            rules, norm.rows(True), lambda batch: [rule_filter(r, c, self.lid) for r in batch]
        )
        if stop == "rule-filter":
            return
        speech = self.store("speech-filter", rules.identity)
        self.transform(speech, rules.rows(True), lambda batch: [speech_filter(r, c) for r in batch])
        if stop == "speech-filter":
            return
        exact = self.store("exact-dedup", speech.identity)
        if not exact.complete:
            self.transform(
                exact,
                exact_deduplicate(speech.rows(True), self.scratch / "exact-dedup"),
                lambda batch: batch,
            )
        if stop == "exact-dedup":
            return
        inference = self.store("scoring", exact.identity)
        if not inference.complete:
            self.scorer = self.scorer or BicleanerScorer(c["bicleaner"], self.scratch / "inference")
            total = exact.counts()["accepted"]
            started = time.monotonic()
            current = 0
            for i, batch in enumerate(chunks(exact.rows(True), c["bicleaner"]["shard_size"])):
                if i < len(inference.manifest["parts"]):
                    continue
                scores = self.scorer.score(batch)
                if len(scores) != len(batch):
                    raise ValueError("Scorer returned wrong number of scores")
                import math

                for row, score in zip(batch, scores):
                    if not math.isfinite(float(score)) or not 0 <= score <= 1:
                        raise ValueError("Scorer returned invalid probability")
                    row.update(bicleaner_score=float(score), stage="scoring")
                inference.write(i, batch)
                current += len(batch)
                rate = current / max(1e-9, time.monotonic() - started)
                done = inference.counts()["accepted"]
                LOG.info(
                    "stage=scoring processed=%d total=%d pairs/sec=%.2f estimated_remaining_seconds=%.1f",
                    done,
                    total,
                    rate,
                    max(0, total - done) / rate,
                )
            metrics = self.scorer.metrics() if hasattr(self.scorer, "metrics") else {}
            inference.finish(metrics)
        score = self.store("score", inference.identity)
        if not score.complete:
            self.transform(
                score,
                rank_candidates(
                    inference.rows(True),
                    c["bicleaner"]["candidate_size"],
                    self.scratch / "candidates",
                ),
                lambda batch: batch,
            )
        if stop == "score":
            return
        dedup = self.store("dedup", score.identity)
        if not dedup.complete:
            self.transform(
                dedup,
                deduplicate(score.rows(True), c["one_side_dedup"], self.scratch / "dedup"),
                lambda batch: batch,
            )
        if stop == "dedup":
            return
        selected = self.store("select", dedup.identity)
        if not selected.complete:
            self.transform(
                selected,
                select_rows(dedup.rows(True), c, self.scratch / "select"),
                lambda batch: batch,
            )
        from .reporting import export_selected

        export_selected(self)
        count = selected.counts()["accepted"]
        if count < c["target_size"]:
            LOG.warning(
                "Selection shortfall: selected=%d target=%d; inspect rejection statistics",
                count,
                c["target_size"],
            )
        if stop == "select":
            return
        from .reporting import audit, statistics

        if stop == "audit":
            audit(self)
        else:
            audit(self)
            statistics(self)
