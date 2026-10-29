"""Bounded-memory I/O and crash-safe, content-verified shard checkpoints."""

import csv
import gzip
import hashlib
import json
import logging
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import __version__
from .schema import SCHEMA

LOG = logging.getLogger(__name__)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
    tmp.replace(path)


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def chunks(rows, size):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def source_identity(c):
    p = Path(c["paths"]["input"]).resolve()
    s = p.stat()
    if c["input"]["sha256"] and file_hash(p) != c["input"]["sha256"]:
        raise ValueError("Input SHA256 differs from configured checksum")
    return {
        "path": str(p),
        "size": s.st_size,
        "mtime_ns": s.st_mtime_ns,
        "sha256": c["input"]["sha256"],
    }


def input_rows(c):
    p = Path(c["paths"]["input"])
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(
            f,
            delimiter=c["input"]["delimiter"],
            quoting=csv.QUOTE_MINIMAL if c["input"]["quoting"] else csv.QUOTE_NONE,
        )
        mapping = c["input"]["columns"]
        if c["input"]["header"]:
            header = next(reader)
            if len(set(header)) != len(header):
                raise ValueError("Duplicate input column names")
            mapping = {k: header.index(v) for k, v in mapping.items()}
        for i, fields in enumerate(reader, start=1):
            expected = c["input"]["expected_columns"]
            if (expected is not None and len(fields) != expected) or len(fields) <= max(
                mapping.values()
            ):
                # Do not echo possibly sensitive source text.
                raise ValueError(f"Invalid column count at record {i}: got {len(fields)}")
            yield {
                "pair_id": f"p{i:012d}",
                "source_line": reader.line_num,
                "ja_raw": fields[mapping["ja"]],
                "en_raw": fields[mapping["en"]],
            }


@contextmanager
def pipeline_lock(root):
    """OS lock released after process death; persistent file is harmless on resume.

    Advisory locking on mounted Drive may be unsupported. In that case abort,
    rather than allow two writers to corrupt checkpoint manifests.
    """
    p = Path(root) / "checkpoints" / "pipeline.lock"
    p.parent.mkdir(parents=True, exist_ok=True)
    f = p.open("a+b")
    try:
        f.seek(0)
        if p.stat().st_size == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            if __import__("os").name == "nt":
                import msvcrt

                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise RuntimeError("Work directory is locked or filesystem locking unavailable") from e
        yield
    finally:
        f.close()


class StageStore:
    """An immutable generation; CURRENT points only to a complete generation.

    Rebuilding a stage never deletes prior data. Force creates a fresh generation,
    naturally invalidating downstream signatures. Partial shards are reused only
    after SHA256 and Arrow footer verification.
    """

    def __init__(self, root, name, settings, upstream, resume=False, force=False):
        self.base = Path(root) / "data" / name
        signature = digest(
            {"version": __version__, "stage": name, "settings": settings, "upstream": upstream}
        )
        self.base.mkdir(parents=True, exist_ok=True)
        active = self.base / "ACTIVE.json"
        previous = read_json(active) if active.exists() else None
        if previous and previous["signature"] == signature and not force:
            if not resume:
                raise FileExistsError(f"{name}: checkpoint exists; use --resume or --force {name}")
            self.path = self.base / previous["generation"]
            self.manifest = read_json(self.path / "manifest.json")
            for part in self.manifest["parts"]:
                self.verify(part)
        else:
            generation = signature[:16] + "-" + uuid.uuid4().hex[:12]
            self.path = self.base / generation
            self.path.mkdir()
            self.manifest = {
                "stage": name,
                "signature": signature,
                "generation": generation,
                "settings": settings,
                "upstream": upstream,
                "complete": False,
                "parts": [],
                "metrics": {},
            }
            self.save()
            atomic_json(active, {"signature": signature, "generation": generation})

    @property
    def complete(self):
        return self.manifest["complete"]

    @property
    def identity(self):
        if not self.complete:
            raise RuntimeError("Incomplete upstream stage")
        return digest(
            {
                "signature": self.manifest["signature"],
                "generation": self.manifest["generation"],
                "parts": self.manifest["parts"],
            }
        )

    def save(self):
        atomic_json(self.path / "manifest.json", self.manifest)

    def verify(self, part):
        p = self.path / part["file"]
        if (
            not p.is_file()
            or file_hash(p) != part["sha256"]
            or pq.ParquetFile(p).metadata.num_rows != part["rows"]
        ):
            raise ValueError(f"Corrupt checkpoint: {p}; rebuild using --force")

    def write(self, index, rows):
        if index < len(self.manifest["parts"]):
            return  # Verified by constructor; deterministic partition boundaries.
        if index != len(self.manifest["parts"]):
            raise ValueError("Checkpoint partitions must be contiguous")
        p = self.path / f"part-{index:06d}.parquet"
        temp = p.with_suffix(".partial")
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), temp, compression="zstd")
        temp.replace(p)
        reasons = Counter(
            reason for row in rows for reason in row.get("reject_reason", "").split(";") if reason
        )
        kept = sum(bool(row["accepted"]) for row in rows)
        self.manifest["parts"].append(
            {
                "file": p.name,
                "sha256": file_hash(p),
                "rows": len(rows),
                "accepted": kept,
                "reasons": dict(reasons),
            }
        )
        self.save()
        LOG.info(
            "stage=%s shard=%d processed=%d accepted=%d rejected=%d checkpoint=%s",
            self.manifest["stage"],
            index,
            len(rows),
            kept,
            len(rows) - kept,
            p,
        )

    def finish(self, metrics=None):
        self.manifest["complete"] = True
        self.manifest["metrics"] = metrics or {}
        self.save()
        atomic_json(
            self.base / "CURRENT.json",
            {"generation": self.path.name, "signature": self.manifest["signature"]},
        )

    def rows(self, accepted_only=False, batch_size=10000):
        for part in self.manifest["parts"]:
            for batch in pq.ParquetFile(self.path / part["file"]).iter_batches(
                batch_size=batch_size
            ):
                for row in batch.to_pylist():
                    if not accepted_only or row["accepted"]:
                        yield row

    def counts(self):
        reasons = Counter()
        for part in self.manifest["parts"]:
            reasons.update(part["reasons"])
        count = sum(p["rows"] for p in self.manifest["parts"])
        kept = sum(p["accepted"] for p in self.manifest["parts"])
        return {
            "stage": self.manifest["stage"],
            "input": count,
            "accepted": kept,
            "rejected": count - kept,
            "reject_rate": (count - kept) / max(1, count),
            "reject_reasons": dict(reasons),
            "metrics": self.manifest["metrics"],
        }
