import pytest

from jparacrawl_filtering.filters import (
    normalize_row,
    normalize_text,
    numeric_consistent,
    rule_filter,
    speech_filter,
)


def row(
    ja="妹と一緒に公園で赤い鳥を観察しました。",
    en="I watched red birds in the park with my sister.",
):
    return normalize_row({"pair_id": "test", "ja_raw": ja, "en_raw": en})


def test_normal_sentence(config, lid):
    r = speech_filter(rule_filter(row(), config, lid), config)
    assert r["accepted"] and r["sentence_like_ja"] and r["sentence_like_en"]


@pytest.mark.parametrize(
    "ja,en,reason",
    [
        (
            "説明はhttps://synthetic.invalidを見てください。",
            "Please see https://synthetic.invalid.",
            "url",
        ),
        ("<div>今日は雨です。</div>", "<div>It is raining today.</div>", "html"),
        ("ホーム ＞ 庭 ＞ 果物", "Home > Garden > Fruit", "navigation"),
        ("著作権、無断転載禁止。", "Copyright. All rights reserved.", "boilerplate"),
        ("処理は if (x > 0) { return x; } です。", "Use if (x > 0) { return x; }.", "code_like"),
        ("計算式は E = mc^2 です。", "The formula is E = mc^2.", "formula_like"),
        ("連絡は hi@synthetic.invalid です。", "Contact hi@synthetic.invalid.", "email"),
        ("説明書は garden.pdf にあります。", "The manual is garden.pdf.", "file_path"),
        ("会場には42個の椅子があります。", "There are 24 chairs at the venue.", "numeric_mismatch"),
    ],
)
def test_rule_rejections(ja, en, reason, config, lid):
    result = rule_filter(row(ja, en), config, lid)
    assert not result["accepted"]
    assert reason in result["reject_reason"].split(";")
    assert result["first_reject_stage"] == "rule-filter"


@pytest.mark.parametrize(
    "ja,en",
    [
        ("「昨日は公園で鳥を見ました。」", '"I watched birds in the park yesterday."'),
        ("（昨日は公園で鳥を見ました。）", "(I watched birds in the park yesterday.)"),
        ("『それは本当に青い鳥ですか？』", "‘Is that really a blue bird?’"),
    ],
)
def test_quoted(ja, en, config):
    assert speech_filter(row(ja, en), config)["accepted"]


def test_fragment_and_bullets(config):
    assert (
        "not_sentence_like"
        in speech_filter(row("公園にある大きな木", "A large tree in the park"), config)[
            "reject_reason"
        ]
    )
    r = speech_filter(row("・青い鳥。\n・赤い鳥。", "• Blue birds.\n• Red birds."), config)
    assert "bullet_like" in r["reject_reason"]
    assert speech_filter(
        row("昨日は公園で鳥を見ました。", "- I watched birds in the park yesterday."), config
    )["bullet_like"]


def test_single_separator_does_not_reject(config, lid):
    r = rule_filter(
        row("昨日の測定では値は3より大きいです。", "The measured value is > 3 today."), config, lid
    )
    assert "navigation" not in r["reject_reason"]


def test_long_boilerplate_discussion(config, lid):
    r = rule_filter(
        row(
            "私たちは新しいウェブサイトを公開する前に、利用規約について十分に説明を受けてから関係者全員で慎重に内容を検討しました。",
            "Before publishing our new website, we carefully discussed the Terms of Service with everyone involved after receiving a detailed explanation.",
        ),
        config,
        lid,
    )
    assert "boilerplate" not in r["reject_reason"]


def test_normalization_and_numbers():
    assert normalize_text("\tＡＢＣ　ガー\x00\u200b\r\nです。  ") == "ABC ガー です。"
    assert numeric_consistent("1,000個、3.14です。", "There are 1000 items and 3.14.")
    assert numeric_consistent("一個です。", "There is 1 item.")
    assert not numeric_consistent("42個です。", "There are 43 items.")
    assert not numeric_consistent("2個と2個です。", "There are 2 items.")


def test_language_and_disable_numbers(config):
    class Wrong:
        def predict(self, text):
            return "fr", 0.7

    result = rule_filter(row(), config, Wrong())
    assert {"lang_mismatch", "lang_low_confidence"} <= set(result["reject_reason"].split(";"))
    config["numbers"]["enabled"] = False
    assert (
        "numeric_mismatch"
        not in rule_filter(
            row("42個の赤い椅子です。", "There are 43 red chairs."), config, Wrong()
        )["reject_reason"]
    )
