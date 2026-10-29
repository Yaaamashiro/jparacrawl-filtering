"""Optional CPU smoke test: stream 310k synthetic candidates and select exactly 300k."""

import argparse
import logging
import time
from pathlib import Path

from jparacrawl_filtering.config import load_config
from jparacrawl_filtering.ranking import select_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=int, default=310000)
    p.add_argument("--target-size", type=int, default=300000)
    p.add_argument("--scratch", type=Path, default=Path(".work/selection-benchmark"))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    c = load_config(root / "configs/default.yaml")
    c["target_size"] = args.target_size

    def rows():
        for i in range(args.pairs):
            n = (5, 20, 40, 80, 120)[i % 5]
            yield {
                "pair_id": f"synthetic-{i:09d}",
                "ja_char_count": n,
                "ja_norm": "合成文。",
                "en_norm": "Synthetic sentence.",
                "bicleaner_score": (args.pairs - i) / max(1, args.pairs),
                "accepted": True,
                "reject_reason": "",
                "first_reject_stage": "",
            }

    start = time.monotonic()
    count = sum(r["accepted"] for r in select_rows(rows(), c, args.scratch))
    assert count == min(args.pairs, args.target_size)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info(
        "Synthetic candidates=%d selected=%d seconds=%.2f",
        args.pairs,
        count,
        time.monotonic() - start,
    )


if __name__ == "__main__":
    main()
