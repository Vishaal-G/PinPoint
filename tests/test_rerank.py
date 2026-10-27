import pandas as pd

from pinpoint.rerank import doc_text, pair_key, rerank

FNS = pd.DataFrame({
    "function_id": ["a.py::f", "a.py::g", "b.py::h", "b.py::k"],
    "path": ["a.py", "a.py", "b.py", "b.py"],
    "qualname": ["f", "g", "h", "k"],
    "source": ["def f(): pass", "def g(): rewrite()", "def h(): pass", "def k(): rewrite()"],
}).set_index("function_id")


def keyword_scorer(query, docs):
    """Stand-in for a cross-encoder: 1 if the doc contains the query word."""
    return [float(query in d) for d in docs]


def test_top_n_reordered_rest_untouched():
    ranking = ["a.py::f", "a.py::g", "b.py::h", "b.py::k"]
    # Only the top 3 are rescored: g moves up; k (4th) stays last even though it would score 1.
    assert rerank(ranking, FNS, "rewrite", keyword_scorer, n=3) == ["a.py::g", "a.py::f", "b.py::h", "b.py::k"]


def test_ties_keep_first_stage_order():
    ranking = ["b.py::h", "a.py::f"]
    assert rerank(ranking, FNS, "nothing", keyword_scorer, n=2) == ranking


def test_header_puts_path_and_name_above_source():
    assert doc_text(FNS.loc["a.py::g"], header=True) == "a.py\ng\ndef g(): rewrite()"
    assert doc_text(FNS.loc["a.py::g"]) == "def g(): rewrite()"


def test_pair_key_depends_on_model_query_and_doc():
    keys = {pair_key("m", "q", "d"), pair_key("m2", "q", "d"), pair_key("m", "q2", "d"), pair_key("m", "q", "d2")}
    assert len(keys) == 4
