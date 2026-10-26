import pandas as pd
import pytest

from pinpoint.fusion import rank_functions_bm25, rrf


def test_rrf_worked_example():
    # A: ranks 1 and 30 -> 1/61 + 1/90 = .0275;  B: 8 and 6 -> 1/68 + 1/66 = .0299;  C: only rank 1 -> .0164
    keyword = ["A"] + [f"k{i}" for i in range(6)] + ["B"]
    vector = ["C"] + [f"v{i}" for i in range(4)] + ["B"] + [f"w{i}" for i in range(23)] + ["A"]
    fused = rrf([keyword, vector])
    assert fused.index("B") < fused.index("A") < fused.index("C")


def test_rrf_missing_item_scores_from_one_list_only():
    assert rrf([["a", "b"], ["b"]]) == ["b", "a"]


def test_rrf_weights_favour_one_list():
    assert rrf([["a", "b"], ["b", "a"]], weights=[1.0, 2.0])[0] == "b"


def test_rrf_dedupes_each_list():
    # With k=0, deduped: a = 1 + 1/3, b = 1/2 + 1 -> b first.
    # Without dedupe a would also collect 1/2 for its repeat and b would sit at rank 3 -> a first.
    assert rrf([["a", "a", "b"], ["b", "c", "a"]], k=0)[0] == "b"


def test_rrf_weights_must_match_lists():
    with pytest.raises(ValueError):
        rrf([["a"], ["b"]], weights=[1.0])


def test_rank_functions_bm25_prefers_matching_identifier():
    fns = pd.DataFrame({
        "function_id": ["a.py::f", "a.py::g", "b.py::h"],
        "source": ["def f(): return parse_config()", "def g(): return 1", "def h(): print('hello')"],
    })
    assert rank_functions_bm25(fns, "crash in parse_config when file is empty")[0] == "a.py::f"
