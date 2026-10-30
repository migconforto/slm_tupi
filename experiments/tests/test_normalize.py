import json

import pytest

from tupi_exp.config import V2_PATH
from tupi_exp.normalize import normalize_tupi, search_tokens


@pytest.mark.parametrize(
    "text, expected",
    [
        ("  abá   ogûatá  ", "abá ogûatá"),
        ("ka’a", "ka'a"),
        ("ka‘a", "ka'a"),
        ("kaʼa", "ka'a"),
        ("ka`a", "ka'a"),
        ("ka´a", "ka'a"),
        ("ô", "ô"),
        ("î", "î"),
    ],
)
def test_safe_rules(text, expected):
    assert normalize_tupi(text) == expected


@pytest.mark.parametrize("text", ["Abá OGÛATÁ", "îagûara, 'ybá; a-b.", "Paranã pu rendupa!"])
def test_case_diacritics_and_inner_punctuation_stay(text):
    assert normalize_tupi(text) == text


def test_idempotent_on_the_whole_dataset():
    with open(V2_PATH, encoding="utf-8") as f:
        rows = json.load(f)
    for row in rows:
        once = normalize_tupi(row["input"])
        assert normalize_tupi(once) == once


def test_aggressive_mode_is_not_implemented():
    with pytest.raises(NotImplementedError):
        normalize_tupi("abá", safe=False)


def test_search_tokens_strip_edges_but_keep_the_glottal_stop():
    assert search_tokens("'ybá, ka'a rupi? (abá-mondá)") == ["'ybá", "ka'a", "rupi", "abá-mondá"]
    assert search_tokens("...") == []
