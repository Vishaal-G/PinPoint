import pandas as pd

from pinpoint import report


def test_table_scores_each_strategy_with_intervals():
    gt = pd.DataFrame({"instance_id": ["a", "b", "c"], "repo": ["r"] * 3,
                       "gold_files": [["x.py"], ["y.py"], ["x.py"]],
                       "gold_functions": [["x.py::f"], ["y.py::g"], []]})        # c: file-level only
    ranked = {"dense_fn": {"a": ["x.py::f"], "b": ["x.py::f", "y.py::g"], "c": ["x.py::f"]},
              "dense_file": {"a": ["x.py"], "b": ["x.py", "y.py"], "c": ["y.py"]}}
    out, scores = report.table(ranked, gt)
    assert out.strategy.tolist() == [report.ROWS[0][2], report.ROWS[4][2]]       # display order, file first
    fn = out[out.level == "fn"].iloc[0]
    assert fn.n == 2 and fn.mrr.startswith("0.750 [")                             # (1 + 1/2) / 2, c skipped
    assert scores[("dense_file", "mrr")] == {"a": 1.0, "b": 0.5, "c": 0.0}
    assert report.markdown(out).splitlines()[1].startswith("| --- |")
