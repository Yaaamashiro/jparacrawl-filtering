"""Configuration loading, strict validation and stage-specific fingerprints."""

import math
from copy import deepcopy
from pathlib import Path

import yaml


def positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} integer")


def validate(c):
    for name in ("seed", "target_size", "batch_size"):
        positive(c[name], name, allow_zero=name == "seed")
    for name in ("candidate_size", "shard_size", "benchmark_pairs", "threads", "timeout_seconds"):
        positive(c["bicleaner"][name], f"bicleaner.{name}")
    if c["bicleaner"]["batch_size"] != "auto":
        positive(c["bicleaner"]["batch_size"], "bicleaner.batch_size")
    if c["bicleaner"]["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("bicleaner.device must be auto, cpu or cuda")
    if c["training_text"] not in ("raw", "normalized"):
        raise ValueError("training_text must be raw or normalized")
    for key in ("accepted_samples", "rejected_samples"):
        positive(c["audit"][key], f"audit.{key}", allow_zero=True)
    for value, name in [
        (c["language_id"]["min_confidence"], "language_id.min_confidence"),
        (c["symbols"]["max_weird_symbol_ratio"], "symbols.max_weird_symbol_ratio"),
    ]:
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    bins = c["length_bins"]
    if not bins:
        raise ValueError("length_bins cannot be empty")
    previous = -1
    total = 0
    for i, b in enumerate(bins):
        positive(b["min"], "bin.min", allow_zero=True)
        if b["min"] != previous + 1:
            raise ValueError("length_bins must be contiguous, ordered and start at zero")
        if b["max"] is None:
            if i != len(bins) - 1:
                raise ValueError("Only the last bin may have max: null")
        else:
            positive(b["max"], "bin.max", allow_zero=True)
            if b["max"] < b["min"]:
                raise ValueError("bin.max must be >= bin.min")
            previous = b["max"]
        ratio = b["ratio"]
        if not isinstance(ratio, (float, int)) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError("bin.ratio must be between zero and one")
        total += ratio
    if bins[-1]["max"] is not None or not 0 < total <= 1.000000001:
        raise ValueError("Bins must cover all lengths; ratios must sum to >0 and <=1")
    inp = c["input"]
    if len(inp["delimiter"]) != 1:
        raise ValueError("input.delimiter must be one character")
    for name in ("ja", "en"):
        col = inp["columns"][name]
        if inp["header"]:
            if not isinstance(col, str):
                raise ValueError("Header column mappings must be names")
        else:
            positive(col, f"input.columns.{name}", allow_zero=True)
            if inp["expected_columns"] is not None and col >= inp["expected_columns"]:
                raise ValueError("Column mapping exceeds expected_columns")
    if inp["columns"]["ja"] == inp["columns"]["en"]:
        raise ValueError("Japanese and English columns must differ")
    length = c["length"]
    for lo, hi in (
        ("min_ja_chars", "max_ja_chars"),
        ("min_en_words", "max_en_words"),
        ("min_char_ratio", "max_char_ratio"),
    ):
        if length[lo] < 0 or length[hi] < length[lo]:
            raise ValueError(f"Invalid length limits: {lo}, {hi}")
    return c


def load_config(path):
    with Path(path).open(encoding="utf-8") as f:
        c = yaml.safe_load(f)
    if not isinstance(c, dict):
        raise ValueError("Configuration must be a mapping")
    return validate(deepcopy(c))


def length_bin(n, bins):
    for i, b in enumerate(bins):
        if n >= b["min"] and (b["max"] is None or n <= b["max"]):
            return i
    raise ValueError(f"No length bin for {n}")
