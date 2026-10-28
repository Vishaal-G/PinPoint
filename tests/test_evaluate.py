import numpy as np
import pandas as pd

from pinpoint import evaluate as ev


def fixture():
    """One issue, three functions in two files; the query vector points at b.py::fix."""
    membership = pd.DataFrame({
        "instance_id": ["i"] * 4, "path": ["a.py", "a.py", "b.py", "tests/test_b.py"],
        "qualname": ["helper", "caller", "fix", "test_fix"], "start_line": [1, 5, 1, 1], "end_line": [3, 8, 4, 2],
        "hash": ["h1", "h2", "h3", "h4"]})
    membership["function_id"] = membership.path + "::" + membership.qualname
    functions = pd.DataFrame({"hash": ["h1", "h2", "h3", "h4"],
                              "source": ["def helper(): pass", "def caller(): fix()", "def fix(): crash", "def test_fix(): fix()"]})
    vectors = {"h1": np.array([1.0, 0, 0]), "h2": np.array([0, 1.0, 0]), "h3": np.array([0, 0, 1.0]), "h4": np.array([0, 0, 1.0])}
    edges = pd.DataFrame({"hash": ["h2"], "callee": ["fix"]})
    issues = pd.DataFrame({"instance_id": ["i"], "problem_statement": ["fix crashes"]})
    gt = pd.DataFrame({"instance_id": ["i"], "repo": ["o/r"], "gold_files": [["a.py"]], "gold_functions": [["a.py::caller"]]})
    return membership, functions, vectors, edges, issues, gt


def test_rankings_scores_and_seeds():
    membership, functions, vectors, edges, issues, gt = fixture()
    cands = ev.candidates_from(membership, functions)
    assert "tests/test_b.py::test_fix" not in set(cands.function_id)   # tests are not candidates
    ranked = ev.rankings(issues, cands, vectors, {"i": np.array([0, 0.6, 0.8])}, edges)
    assert ranked["dense_fn"]["i"] == ["b.py::fix", "a.py::caller", "a.py::helper"]
    assert ranked["dense_file"]["i"] == ["b.py", "a.py"]
    assert ranked["graph_fn"]["i"][:2] == ["b.py::fix", "a.py::caller"]  # the caller of the top hit stays close
    table = ev.results_table(ranked, gt).set_index(["strategy", "repo"])
    assert table.loc[("dense_fn", "all"), "mrr"] == 0.5 and table.loc[("dense_file", "o/r"), "acc@1"] == 0.0
    seeds = ev.seeds_table(ranked, cands, k=2)
    assert seeds[["rank", "function_id", "start_line"]].values.tolist() == [[1, "b.py::fix", 1], [2, "a.py::caller", 5]]
