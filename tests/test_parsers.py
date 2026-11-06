"""Response parsers of the LLM judges."""

import math

import pytest

from evaluation.llm_judges import granularity_from_response, parse_coherence, parse_true_false
from source.mine_eval import _parse_judge


@pytest.mark.parametrize("response, expected", [
    ("true", True), ("false", False), ("True.", True), ("'false'", False), ("`true`", True),
    ("The answer is true", True), ("The answer is false.", False),
    ('{"factual": true}', True), ("Yes", True), ("no", False),
    ("", False), (None, False), ("unsure", False),
    ("true, not false", True),  # the first explicit token decides
])
def test_parse_true_false(response, expected):
    assert parse_true_false(response) is expected


@pytest.mark.parametrize("response, expected", [
    ("coherent", True), ("incoherent", False), ("Coherent.", True), ("Incoherent.", False),
    ("The triple is incoherent", False), ("It is coherent", True),
    ("", False), (None, False), ("neither", False),
])
def test_parse_coherence(response, expected):
    assert parse_coherence(response) is expected


@pytest.mark.parametrize("response, n_sub", [
    ("N/A (already atomic)", 0),                       # no brackets
    ('["a", "r", "b"]', 0),                            # one pair: the triple itself
    ('["a", "r", "b"]\n["a", "r", "c"]', 2),
    ('["a","r","b"] ["a","r","c"] ["a","r","d"]', 3),
    ('["a", "r", "b"', 0),                             # unbalanced: min of [ and ]
])
def test_granularity_counts_bracket_pairs(response, n_sub):
    assert granularity_from_response(response) == pytest.approx(math.exp(-n_sub))


@pytest.mark.parametrize("response, expected", [
    ("1", 1), ("0", 0), (" 1\n", 1), ("Answer: 0", 0), ("yes", 1), ("no", 0), ("", 0),
])
def test_mine_judge_parser(response, expected):
    assert _parse_judge(response) == expected
