from copy import deepcopy

import pytest

from jparacrawl_filtering.config import validate
from jparacrawl_filtering.ranking import deduplicate, rank_candidates, select_rows


def make(i, ja=None, en=None, length=40, score=None):
    ja, en = ja or f"日本語{i}。", en or f"English sentence {i}."
    return {
        "pair_id": f"p{i:06d}",
        "ja_norm": ja,
        "en_norm": en,
        "ja_raw": ja,
        "en_raw": en,
        "ja_char_count": length,
        "bicleaner_score": score if score is not None else 1 - i / 1000,
        "accepted": True,
        "reject_reason": "",
        "first_reject_stage": "",
    }


def test_dedup_ranks_and_reasons(tmp_path):
    rows = [
        make(4, "同じ。", "Same.", score=0.5),
        make(1, "同じ。", "Same.", score=0.99),
        make(2, "同じ。", "Other.", score=0.98),
        make(3, "別。", "Same.", score=0.97),
        make(5, "別。", "Other.", score=0.96),
    ]
    result = list(deduplicate(rows, True, tmp_path))
    assert [r["pair_id"] for r in result if r["accepted"]] == ["p000001", "p000005"]
    assert "duplicate_pair" in result[-1]["reject_reason"]
    assert "duplicate_ja" in result[1]["reject_reason"]
    assert "duplicate_en" in result[2]["reject_reason"]
    assert sum(r["accepted"] for r in deduplicate(rows, False, tmp_path)) == 4


def test_candidate_ranking_deterministic_ties(tmp_path):
    result = list(rank_candidates([make(2, score=0.9), make(1, score=0.9), make(3)], 2, tmp_path))
    assert [r["pair_id"] for r in result if r["accepted"]] == ["p000003", "p000001"]
    assert result[-1]["reject_reason"] == "quality_rank"


def test_length_quota_and_refill(config, tmp_path):
    config["target_size"] = 100
    rows = [
        make(i, length=n)
        for i, n in enumerate(([5] * 20 + [20] * 30 + [40] * 70 + [80] * 50 + [120] * 30), 1)
    ]
    result = list(select_rows(rows, config, tmp_path))
    selected = [r for r in result if r["accepted"]]
    assert len(selected) == 100
    assert [sum(r["length_bin"] == i for r in selected) for i in range(5)] == [2, 15, 40, 33, 10]
    assert all(r["ja"] == r["ja_norm"] for r in selected)
    sparse = [make(i, length=40) for i in range(125)]
    assert sum(r["accepted"] for r in select_rows(sparse, config, tmp_path)) == 100
    assert sum(r["accepted"] for r in select_rows(sparse[:12], config, tmp_path)) == 12


def test_bins_with_zero_ratio_disabled(config, tmp_path):
    config["length_bins"][0]["ratio"] = 0
    rows = [make(i, length=5) for i in range(30)]
    assert not any(r["accepted"] for r in select_rows(rows, config, tmp_path))


@pytest.mark.parametrize("bad", [float("nan"), -0.2, 1.2])
def test_config_invalid_ratio(config, bad):
    config["length_bins"][0]["ratio"] = bad
    with pytest.raises(ValueError):
        validate(config)


def test_config_validation(config):
    c = deepcopy(config)
    c["length_bins"][0]["ratio"] = 0.9
    with pytest.raises(ValueError):
        validate(c)
    config["length_bins"][1]["min"] = 8
    with pytest.raises(ValueError):
        validate(config)
