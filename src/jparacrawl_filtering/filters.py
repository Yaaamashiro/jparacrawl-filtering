"""CPU-only, explainable filters. All samples in tests are original synthetic text."""

import re
import unicodedata
from collections import Counter
from decimal import Decimal

from .schema import reject

URL = re.compile(r"https?://|www\.", re.I)
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.I)
HTML = re.compile(r"</?[A-Za-z][^>]*>|&(?:[A-Za-z]+|#\d+|#x[\da-f]+);", re.I)
FILE_PATH = re.compile(
    r"\b[\w.-]+\.(?:pdf|jpe?g|png|gif|svg|docx?|xlsx?|pptx?|zip|exe|mp[34]|html?)\b"
    r"|\b[A-Za-z]:\\|(?:^|\s)/(?:usr|home|var|etc|tmp|content)/",
    re.I,
)
SEPARATOR = re.compile(r"[>＞|｜»›]")
BULLET = re.compile(r"(?:^|\n)\s*(?:[・•■●▪◆]\s*|[-*]\s+)")
CODE = re.compile(r"\b(?:if|for|while)\s*\([^)]*\)\s*\{|\breturn\s+[^;]+;|[{}].*[;{}]", re.S)
FORMULA = re.compile(r"[∫∑√]|\b\w+\s*=\s*[\w(]+(?:\s*[+*/^]\s*[\w()]+)+")
NUMBERS = re.compile(r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?|(?<=[^\x00-\x7f])\d+(?:\.\d+)?")
WORDS = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*|\d+(?:\.\d+)?")
JA_END = re.compile(r"""[。！？!?][」』）)】〕〉》\]”’"']*$""")
EN_END = re.compile(r"""[.!?][”’"')\]）】」』}]*$""")
NORMAL_PUNCT = set("。、！？,.?!'\"-–—:;()[]{}「」『』（）【】…・、%％/\\")


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    # Newlines become spaces, other controls/formatting disappear. Japanese letters,
    # kana combining marks and long vowels are preserved by Unicode normalization.
    text = "".join(
        " " if ch.isspace() else ch
        for ch in text
        if ch.isspace() or unicodedata.category(ch) not in ("Cc", "Cf", "Cs")
    )
    return re.sub(r"\s+", " ", text).strip()


def weird_symbol_ratio(text):
    chars = [ch for ch in text if not ch.isspace()]
    weird = sum(
        unicodedata.category(ch)[0] in ("S", "P", "C") and ch not in NORMAL_PUNCT for ch in chars
    )
    return weird / max(1, len(chars))


def numbers(text):
    return Counter(Decimal(x.replace(",", "")) for x in NUMBERS.findall(text))


def numeric_consistent(ja, en, require_both=True):
    a, b = numbers(ja), numbers(en)
    return True if require_both and (not a or not b) else a == b


def normalize_row(row):
    ja, en = normalize_text(row["ja_raw"]), normalize_text(row["en_raw"])
    row.update(
        ja_norm=ja,
        en_norm=en,
        ja_char_count=len(ja),
        en_char_count=len(en),
        en_word_count=len(WORDS.findall(en)),
        length_ratio=len(ja) / max(1, len(en)),
        accepted=True,
        reject_reason="",
        first_reject_stage="",
        stage="normalize",
    )
    row["bullet_like"] = bool(BULLET.search(row["ja_raw"]) or BULLET.search(row["en_raw"]))
    return reject(row, ["empty_text"] if not ja or not en else [], "normalize")


class FastTextLanguageID:
    def __init__(self, model_path):
        from pathlib import Path

        if not Path(model_path).is_file():
            raise FileNotFoundError("Language ID model missing; run download-lid --config CONFIG")
        try:
            import fasttext
        except ImportError as e:
            raise RuntimeError("Install the lid extra: pip install '.[lid]'") from e
        self.model = fasttext.load_model(str(model_path))

    def predict(self, text):
        labels, scores = self.model.predict(text.replace("\n", " "), k=1)
        return labels[0].removeprefix("__label__"), float(scores[0])


def rule_filter(row, c, lid):
    ja, en = row["ja_norm"], row["en_norm"]
    reasons = []
    row["ja_lang"], row["ja_lang_score"] = lid.predict(ja)
    row["en_lang"], row["en_lang_score"] = lid.predict(en)
    if row["ja_lang"] != "ja" or row["en_lang"] != "en":
        reasons.append("lang_mismatch")
    if min(row["ja_lang_score"], row["en_lang_score"]) < c["language_id"]["min_confidence"]:
        reasons.append("lang_low_confidence")
    limits = c["length"]
    if (
        row["ja_char_count"] < limits["min_ja_chars"]
        or row["en_word_count"] < limits["min_en_words"]
    ):
        reasons.append("too_short")
    if (
        row["ja_char_count"] > limits["max_ja_chars"]
        or row["en_word_count"] > limits["max_en_words"]
    ):
        reasons.append("too_long")
    if not limits["min_char_ratio"] <= row["length_ratio"] <= limits["max_char_ratio"]:
        reasons.append("length_ratio")
    for name, pattern in (("url", URL), ("email", EMAIL), ("html", HTML), ("file_path", FILE_PATH)):
        row[f"has_{name}"] = bool(pattern.search(ja) or pattern.search(en))
        if row[f"has_{name}"]:
            reasons.append(name)
    counts = [len(SEPARATOR.findall(t)) for t in (ja, en)]
    densities = [n / max(1, len(t)) for n, t in zip(counts, (ja, en))]
    row["navigation_separator_count"] = max(counts)
    row["navigation_score"] = max(densities)
    if any(
        n >= c["navigation"]["min_separators"] and d >= c["navigation"]["min_density"]
        for n, d in zip(counts, densities)
    ):
        reasons.append("navigation")
    bp = c["boilerplate"]
    keyword = [any(k.casefold() in t.casefold() for k in bp["keywords"]) for t in (ja, en)]
    short = [len(ja) <= bp["max_ja_chars"], row["en_word_count"] <= bp["max_en_words"]]
    bp_hits = [
        hit and (small or "©" in raw)
        for hit, small, raw in zip(keyword, short, (row["ja_raw"], row["en_raw"]))
    ]
    row["boilerplate_score"] = 1.0 if any(bp_hits) else (0.5 if any(keyword) else 0.0)
    if any(bp_hits):
        reasons.append("boilerplate")
    for side, text in (("ja", ja), ("en", en)):
        row[f"weird_symbol_ratio_{side}"] = weird_symbol_ratio(text)
    if (
        max(row["weird_symbol_ratio_ja"], row["weird_symbol_ratio_en"])
        > c["symbols"]["max_weird_symbol_ratio"]
    ):
        reasons.append("symbol_ratio")
    if CODE.search(ja) or CODE.search(en):
        reasons.append("code_like")
    if FORMULA.search(ja) or FORMULA.search(en):
        reasons.append("formula_like")
    row["numeric_consistent"] = numeric_consistent(ja, en, c["numbers"]["require_both_sides"])
    if c["numbers"]["enabled"] and not row["numeric_consistent"]:
        reasons.append("numeric_mismatch")
    return reject(row, reasons, "rule-filter")


def speech_filter(row, c):
    row["sentence_like_ja"] = bool(JA_END.search(row["ja_norm"]))
    row["sentence_like_en"] = bool(EN_END.search(row["en_norm"]))
    reasons = []
    if c["sentence"]["require_terminal_punctuation"] and not (
        row["sentence_like_ja"] and row["sentence_like_en"]
    ):
        reasons.append("not_sentence_like")
    if c["sentence"]["reject_bullets"] and row["bullet_like"]:
        reasons.append("bullet_like")
    return reject(row, reasons, "speech-filter")
