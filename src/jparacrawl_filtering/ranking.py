"""Disk-backed ranking, greedy one-side deduplication and length quotas."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .config import length_bin
from .schema import reject


def exact_deduplicate(rows, scratch):
    """Keep the first normalized pair; leave one-sided alternatives for scoring.

    The disk-backed set is rebuilt on resume while the caller replays and skips
    already checkpointed output batches. No text hash collisions are possible.
    """
    p = Path(scratch) / "exact-pairs.sqlite"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.unlink(missing_ok=True)
    db = sqlite3.connect(p)
    try:
        db.execute("PRAGMA cache_size=-65536")
        db.execute("CREATE TABLE seen (ja TEXT, en TEXT, PRIMARY KEY (ja,en)) WITHOUT ROWID")
        for row in rows:
            inserted = db.execute(
                "INSERT OR IGNORE INTO seen VALUES (?,?)", (row["ja_norm"], row["en_norm"])
            ).rowcount
            yield reject(row, [] if inserted else ["duplicate_pair"], "exact-dedup")
    finally:
        db.close()
        p.unlink(missing_ok=True)


@contextmanager
def ranked_database(rows, scratch, bins=None):
    """SQLite bounds its cache; payloads and sort/index state spill to disk."""
    p = Path(scratch) / "ranking.sqlite"
    p.parent.mkdir(parents=True, exist_ok=True)
    # Disposable, project-owned scratch. Old DB is rebuilt after interruption.
    p.unlink(missing_ok=True)
    db = sqlite3.connect(p)
    try:
        db.execute("PRAGMA cache_size=-65536")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute(
            "CREATE TABLE pairs (id TEXT PRIMARY KEY, score REAL, bin INTEGER, payload TEXT)"
        )

        def records():
            for row in rows:
                b = length_bin(row["ja_char_count"], bins) if bins else -1
                yield row["pair_id"], row["bicleaner_score"], b, json.dumps(row, ensure_ascii=False)

        db.executemany("INSERT INTO pairs VALUES (?,?,?,?)", records())
        db.execute("CREATE INDEX quality_order ON pairs(score DESC, id ASC)")
        db.execute("CREATE INDEX bin_order ON pairs(bin, score DESC, id ASC)")
        db.commit()
        yield db
    finally:
        db.close()
        p.unlink(missing_ok=True)


def rank_candidates(rows, candidate_size, scratch):
    with ranked_database(rows, scratch) as db:
        for rank, (payload,) in enumerate(
            db.execute("SELECT payload FROM pairs ORDER BY score DESC, id ASC")
        ):
            row = json.loads(payload)
            yield reject(row, ["quality_rank"] if rank >= candidate_size else [], "score")


def deduplicate(rows, one_side, scratch):
    with ranked_database(rows, scratch) as db:
        db.executescript("""
            CREATE TABLE seen_pair (ja TEXT, en TEXT, PRIMARY KEY (ja,en));
            CREATE TABLE seen_ja (text TEXT PRIMARY KEY);
            CREATE TABLE seen_en (text TEXT PRIMARY KEY);
        """)
        for (payload,) in db.execute("SELECT payload FROM pairs ORDER BY score DESC, id ASC"):
            row = json.loads(payload)
            ja, en = row["ja_norm"], row["en_norm"]
            reasons = []
            if db.execute("SELECT 1 FROM seen_pair WHERE ja=? AND en=?", (ja, en)).fetchone():
                reasons.append("duplicate_pair")
            if one_side:
                if db.execute("SELECT 1 FROM seen_ja WHERE text=?", (ja,)).fetchone():
                    reasons.append("duplicate_ja")
                if db.execute("SELECT 1 FROM seen_en WHERE text=?", (en,)).fetchone():
                    reasons.append("duplicate_en")
            if not reasons:
                db.execute("INSERT INTO seen_pair VALUES (?,?)", (ja, en))
                db.execute("INSERT OR IGNORE INTO seen_ja VALUES (?)", (ja,))
                db.execute("INSERT OR IGNORE INTO seen_en VALUES (?)", (en,))
            yield reject(row, reasons, "dedup")


def select_rows(rows, c, scratch):
    bins = c["length_bins"]
    with ranked_database(rows, scratch, bins) as db:
        db.execute("CREATE TABLE selected (id TEXT PRIMARY KEY)")
        for i, b in enumerate(bins):
            quota = int(c["target_size"] * b["ratio"])
            db.execute(
                """INSERT INTO selected SELECT id FROM pairs WHERE bin=?
                          ORDER BY score DESC, id ASC LIMIT ?""",
                (i, quota),
            )
        count = db.execute("SELECT COUNT(*) FROM selected").fetchone()[0]
        # Fractional rounding, sum(ratios)<1 and underfilled bins all flow here.
        allowed = [i for i, b in enumerate(bins) if b["ratio"] > 0]
        placeholders = ",".join("?" for _ in allowed)
        db.execute(
            f"""INSERT INTO selected SELECT id FROM pairs
                       WHERE bin IN ({placeholders}) AND id NOT IN (SELECT id FROM selected)
                       ORDER BY score DESC, id ASC LIMIT ?""",
            (*allowed, max(0, c["target_size"] - count)),
        )
        cursor = db.execute("""SELECT payload, bin, selected.id IS NOT NULL FROM pairs
                               LEFT JOIN selected USING(id) ORDER BY score DESC, pairs.id ASC""")
        for payload, b, selected in cursor:
            row = json.loads(payload)
            row["length_bin"] = b
            suffix = "raw" if c["training_text"] == "raw" else "norm"
            row["ja"], row["en"] = row[f"ja_{suffix}"], row[f"en_{suffix}"]
            yield reject(row, [] if selected else ["length_quota"], "select")
