import pytest

from pinpoint.text import split_identifier, tokenize


def test_snake_case():
    assert split_identifier("parse_patch_file") == ["parse", "patch", "file"]


def test_camel_case():
    assert split_identifier("getUserName") == ["get", "user", "name"]


def test_acronyms():
    assert split_identifier("HTTPError") == ["http", "error"]
    assert split_identifier("parseXMLFile") == ["parse", "xml", "file"]
    assert split_identifier("AssertionRewriter") == ["assertion", "rewriter"]


def test_dunder_and_private():
    assert split_identifier("__init__") == ["init"]
    assert split_identifier("_pytest") == ["pytest"]


CODE = "def is_rewrite_disabled(self, doc): return 'PYTEST_DONT_REWRITE' in doc  # x = 1"


def test_tokenize_whole():
    assert tokenize(CODE, "whole") == [
        "def", "is_rewrite_disabled", "self", "doc", "return", "pytest_dont_rewrite", "in", "doc",
    ]


def test_tokenize_split():
    assert tokenize(CODE, "split") == [
        "def", "is", "rewrite", "disabled", "self", "doc", "return",
        "pytest", "dont", "rewrite", "in", "doc",
    ]


def test_tokenize_both_keeps_whole_and_parts_without_repeating_simple_words():
    assert tokenize("is_rewrite_disabled self", "both") == [
        "is_rewrite_disabled", "is", "rewrite", "disabled", "self",
    ]


def test_tokenize_prose_drops_punctuation_numbers_and_single_letters():
    assert tokenize("Rewrite fails when a number (e.g. 42) is first!", "split") == [
        "rewrite", "fails", "when", "number", "is", "first",
    ]


def test_tokenize_unknown_mode():
    with pytest.raises(ValueError):
        tokenize("x", "words")
