"""Official Bicleaner-AI CLI adapter; no TensorFlow import in this process."""

import logging
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

LOG = logging.getLogger(__name__)


def cuda_available():
    if os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
        return False
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        return False


class BicleanerScorer:
    def __init__(self, config, scratch):
        self.c = config
        self.scratch = Path(scratch)
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if config["device"] != "cpu" and cuda_available() else "cpu"
        if self.device == "cpu" and config["device"] != "cpu":
            LOG.warning("CUDA unavailable; using Bicleaner CPU fallback")
        self.batch_size = (
            (32 if self.device == "cuda" else 8)
            if config["batch_size"] == "auto"
            else config["batch_size"]
        )
        self.elapsed = 0.0
        self.processed = 0
        self.next_benchmark = config["benchmark_pairs"]

    def score(self, rows):
        if not rows:
            return []
        executable = shutil.which(self.c["executable"])
        if not executable:
            raise RuntimeError("Bicleaner CLI missing; install '.[scoring]' in Linux/Colab")
        inp = self.scratch / "inference-input.tsv"
        out = self.scratch / "inference-scores.txt"
        # The model is en->ja: do not reverse the pair just because the project is ja-en.
        with inp.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(row["en_norm"] + "\t" + row["ja_norm"] + "\n")
        started = time.monotonic()
        try:
            while True:
                out.unlink(missing_ok=True)
                env = os.environ.copy()
                env.update(
                    PYTHONIOENCODING="utf-8",
                    BICLEANER_AI_THREADS=str(self.c["threads"]),
                    TF_FORCE_GPU_ALLOW_GROWTH="true",
                    TOKENIZERS_PARALLELISM="false",
                )
                if self.device == "cpu":
                    env["CUDA_VISIBLE_DEVICES"] = "-1"
                command = [
                    executable,
                    str(inp),
                    str(out),
                    self.c["model"],
                    "--scol",
                    "1",
                    "--tcol",
                    "2",
                    "--score_only",
                    "--disable_hardrules",
                    "--batch_size",
                    str(self.batch_size),
                    "--quiet",
                ]
                if self.device == "cuda":
                    command.append("--require_gpu")
                if self.c["offline"]:
                    command.append("--offline")
                if self.c["calibrated"]:
                    command.append("--calibrated")
                # Bicleaner stderr may contain source sentences; never forward it to logs.
                try:
                    proc = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        env=env,
                        timeout=self.c["timeout_seconds"],
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    raise RuntimeError(
                        "Bicleaner timed out; completed shards are safe to resume"
                    ) from None
                if proc.returncode == 0:
                    break
                error = proc.stderr.lower()
                oom = any(
                    x in error
                    for x in (
                        "out of memory",
                        "resourceexhausted",
                        "oom",
                        "cuda_error_out_of_memory",
                    )
                )
                if oom and self.batch_size > 1:
                    self.batch_size = max(1, self.batch_size // 2)
                    LOG.warning("Bicleaner memory retry: batch_size=%d", self.batch_size)
                    continue
                if self.device == "cuda":
                    self.device = "cpu"
                    self.batch_size = min(self.batch_size, 8)
                    LOG.warning("Bicleaner GPU execution failed; retrying on CPU")
                    continue
                raise RuntimeError(
                    f"Bicleaner failed (exit={proc.returncode}); check model/dependency installation"
                )
            scores = []
            with out.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        score = float(line.strip())
                    except ValueError:
                        raise ValueError(
                            "Bicleaner output must contain one numeric score per row"
                        ) from None
                    if not math.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError("Bicleaner returned an invalid probability")
                    scores.append(score)
            if len(scores) != len(rows):
                raise ValueError("Bicleaner score count differs from input count")
            self.elapsed += time.monotonic() - started
            self.processed += len(rows)
            if self.processed >= self.next_benchmark:
                LOG.info(
                    "Bicleaner benchmark processed=%d pairs/sec=%.2f device=%s batch_size=%d",
                    self.processed,
                    self.processed / self.elapsed,
                    self.device,
                    self.batch_size,
                )
                self.next_benchmark += self.c["benchmark_pairs"]
            return scores
        finally:
            inp.unlink(missing_ok=True)
            out.unlink(missing_ok=True)

    def metrics(self):
        return {
            "processed_this_run": self.processed,
            "elapsed_seconds": self.elapsed,
            "pairs_per_second": self.processed / max(self.elapsed, 1e-9),
            "device": self.device,
            "batch_size": self.batch_size,
        }
