"""Stable Arrow types, including empty shards and future audio measurements."""

import pyarrow as pa

STRINGS = """pair_id ja_raw en_raw ja_norm en_norm ja_lang en_lang reject_reason
first_reject_stage stage ja en""".split()
INTS = """source_line ja_char_count en_char_count en_word_count navigation_separator_count
length_bin""".split()
FLOATS = """ja_lang_score en_lang_score length_ratio navigation_score boilerplate_score
weird_symbol_ratio_ja weird_symbol_ratio_en bicleaner_score estimated_duration actual_duration""".split()
BOOLS = """has_url has_email has_html has_file_path sentence_like_ja sentence_like_en
numeric_consistent bullet_like accepted""".split()
SCHEMA = pa.schema(
    [(s, pa.string()) for s in STRINGS]
    + [(s, pa.int64()) for s in INTS]
    + [(s, pa.float64()) for s in FLOATS]
    + [(s, pa.bool_()) for s in BOOLS]
)


def reject(row, reasons, stage):
    if reasons:
        row.update(
            accepted=False, reject_reason=";".join(dict.fromkeys(reasons)), first_reject_stage=stage
        )
    row["stage"] = stage
    return row
