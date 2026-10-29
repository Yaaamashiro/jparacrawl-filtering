"""Offline wiring demonstration. Scores/LID are test doubles, NEVER research results."""

import argparse
import csv
import json
import logging
import re
from pathlib import Path

from jparacrawl_filtering.config import load_config
from jparacrawl_filtering.pipeline import Pipeline


class SyntheticLanguageID:
    def predict(self, text):
        return ("ja" if re.search(r"[ぁ-んァ-ヶ一-龯]", text) else "en"), 0.99


class SyntheticScorer:
    def score(self, rows):
        return [0.99 - (int(r["pair_id"][1:]) % 20) * 0.01 for r in rows]


def prepare(work):
    root = Path(__file__).resolve().parents[1]
    work = Path(work).resolve()
    raw = work / "raw" / "synthetic.tsv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        samples = json.loads((root / "tests/fixtures/synthetic.json").read_text(encoding="utf-8"))
        with raw.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            for row in samples:
                writer.writerow(
                    ["synthetic.invalid", "synthetic.invalid", "0.5", row["en"], row["ja"]]
                )
    c = load_config(root / "configs/default.yaml")
    c["paths"].update(input=str(raw), work_dir=str(work), scratch_dir=str(work / ".scratch"))
    c["input"]["quoting"] = True
    c.update(target_size=4, batch_size=3)
    c["bicleaner"].update(candidate_size=100, shard_size=3, device="cpu")
    c["audit"].update(accepted_samples=500, rejected_samples=200)
    return c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(".work/synthetic-demo"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.warning("SYNTHETIC DEMO: injected language labels and scores; not Bicleaner validation")
    Pipeline(
        prepare(args.work_dir),
        resume=args.resume,
        lid=SyntheticLanguageID(),
        scorer=SyntheticScorer(),
    ).execute()


if __name__ == "__main__":
    main()
