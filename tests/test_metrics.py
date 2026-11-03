import numpy as np
import pytest

from pinpoint.metrics import (
    accuracy_at_k,
    dedupe,
    evaluate,
    files_from_functions,
    recall_at_k,
    reciprocal_rank,
)


def ranking_with(placed: dict[str, int], length: int = 40) -> list[str]:
    """A ranking of filler ids with the given ids at the given 1-based positions."""
    ranking = [f"other{i}" for i in range(1, length + 1)]
    for id_, pos in placed.items():
        ranking[pos - 1] = id_
    return ranking


# The worked example from the lesson.
A = ranking_with({"run": 2})
B = ranking_with({"call": 1, "setup": 5})
C = ranking_with({"getbasetemp": 37})
RANKINGS = {"A": A, "B": B, "C": C}
GOLD = {"A": {"run"}, "B": {"setup", "call"}, "C": {"getbasetemp"}}


def test_accuracy_at_k_per_issue():
    assert [accuracy_at_k(RANKINGS[i], GOLD[i], 1) for i in "ABC"] == [0, 1, 0]
    assert [accuracy_at_k(RANKINGS[i], GOLD[i], 5) for i in "ABC"] == [1, 1, 0]


def test_recall_at_k_per_issue():
    assert [recall_at_k(RANKINGS[i], GOLD[i], 1) for i in "ABC"] == [0, 0.5, 0]
    assert [recall_at_k(RANKINGS[i], GOLD[i], 5) for i in "ABC"] == [1, 1, 0]


def test_reciprocal_rank_per_issue():
    assert [reciprocal_rank(RANKINGS[i], GOLD[i]) for i in "ABC"] == [1 / 2, 1, 1 / 37]


def test_evaluate_matches_the_lesson():
    r = evaluate(RANKINGS, GOLD, ks=(1, 5))
    assert r["acc@1"] == pytest.approx(1 / 3)
    assert r["acc@5"] == pytest.approx(2 / 3)
    assert r["recall@1"] == pytest.approx(0.5 / 3)
    assert r["recall@5"] == pytest.approx(2 / 3)
    assert r["mrr"] == pytest.approx((1 / 2 + 1 + 1 / 37) / 3)  # 0.51
    assert r["n"] == 3


def test_not_found_scores_zero():
    assert accuracy_at_k(["x", "y"], {"gold"}, 5) == 0
    assert recall_at_k(["x", "y"], {"gold"}, 5) == 0
    assert reciprocal_rank(["x", "y"], {"gold"}) == 0


def test_k_larger_than_ranking():
    assert accuracy_at_k(["x", "gold"], {"gold"}, 100) == 1


def test_duplicates_do_not_take_top_k_slots():
    ranking = ["x", "x", "x", "x", "gold"]
    assert dedupe(ranking) == ["x", "gold"]
    assert accuracy_at_k(ranking, {"gold"}, 2) == 1
    assert reciprocal_rank(ranking, {"gold"}) == 1 / 2


def test_recall_with_no_gold_raises():
    with pytest.raises(ValueError):
        recall_at_k(["x"], set(), 5)


def test_evaluate_skips_issues_without_gold():
    r = evaluate({"A": A, "D": ["x"]}, {"A": {"run"}, "D": []}, ks=(1,))
    assert r["n"] == 1
    assert r["mrr"] == 1 / 2


def test_evaluate_missing_ranking_scores_zero():
    r = evaluate({"A": A}, {"A": {"run"}, "B": {"call"}}, ks=(5,))
    assert r["n"] == 2
    assert r["acc@5"] == 0.5


def test_files_from_functions():
    ranking = ["a.py::f", "a.py::g", "b.py::Cls.m", "a.py::h", "c.py::k"]
    assert files_from_functions(ranking) == ["a.py", "b.py", "c.py"]


def test_evaluate_accepts_numpy_gold_from_parquet():
    r = evaluate({"A": A}, {"A": np.array(["run"]), "D": np.array([], dtype=object)}, ks=(5,))
    assert r["n"] == 1


def test_per_issue_matches_evaluate():
    from pinpoint.metrics import per_issue

    rankings = {"a": ["x", "g"], "b": ["g"], "c": ["y"]}
    gold = {"a": ["g"], "b": ["g"], "c": ["g"], "d": []}
    rr = per_issue(rankings, gold)
    assert rr == {"a": 0.5, "b": 1.0, "c": 0.0}                       # d has no gold: skipped
    assert sum(rr.values()) / len(rr) == pytest.approx(evaluate(rankings, gold)["mrr"])
    assert per_issue(rankings, gold, "acc@1") == {"a": 0.0, "b": 1.0, "c": 0.0}


def test_bootstrap_ci_brackets_the_mean_and_shrinks_with_n():
    from pinpoint.metrics import bootstrap_ci

    small = {str(i): float(i % 2) for i in range(20)}
    big = {str(i): float(i % 2) for i in range(2000)}
    m, lo, hi = bootstrap_ci(small)
    assert m == 0.5 and lo < 0.5 < hi
    _, blo, bhi = bootstrap_ci(big)
    assert bhi - blo < (hi - lo) / 5                                  # width ~ 1/sqrt(n): 10x n -> ~1/10 width
    assert bootstrap_ci({"a": 1.0, "b": 1.0}) == (1.0, 1.0, 1.0)      # no spread, no interval


def test_paired_bootstrap_sees_a_consistent_small_gain():
    from pinpoint.metrics import bootstrap_ci, paired_bootstrap

    hard = {str(i): (i % 10) / 10 for i in range(100)}                # issues vary a lot in difficulty
    better = {i: v + 0.05 for i, v in hard.items()}                   # +0.05 on every issue
    diff, lo, hi, p = paired_bootstrap(better, hard)
    assert diff == pytest.approx(0.05) and lo == pytest.approx(0.05) and p == 0.0
    _, alo, ahi = bootstrap_ci(better)
    _, blo, bhi = bootstrap_ci(hard)
    assert alo < bhi                                                  # unpaired intervals overlap anyway
