import numpy as np
import pandas as pd

from pinpoint.dense import rank_files, rank_functions, search

CHUNKS = pd.DataFrame({
    "path":       ["a.py", "a.py", "b.py"],
    "start_line": [1, 40, 1],
    "end_line":   [45, 80, 30],
})


def test_search_is_dot_product_of_unit_vectors():
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    assert np.allclose(search(vecs, np.array([0.6, 0.8])), [0.6, 0.8, 1.0])


def test_file_score_is_its_best_chunk():
    # a.py's chunks score .2 and .9; b.py's single chunk scores .5. Max wins, not sum or mean.
    assert rank_files(CHUNKS, np.array([0.2, 0.9, 0.5])) == ["a.py", "b.py"]


def test_function_gets_best_overlapping_chunk_and_ties_keep_file_order():
    functions = pd.DataFrame({
        "path":        ["a.py", "a.py", "a.py", "b.py", "tests/t.py"],
        "start_line":  [50, 2, 42, 3, 1],
        "end_line":    [60, 10, 44, 9, 5],
        "function_id": ["a.py::late", "a.py::early", "a.py::edge", "b.py::f", "tests/t.py::test"],
    })
    ranking = rank_functions(CHUNKS, np.array([0.2, 0.9, 0.5]), functions)
    # edge (42-44) overlaps both a.py chunks, so it takes .9 and ties with late; file order breaks it.
    # tests/t.py has no chunks, so it is not a candidate.
    assert ranking == ["a.py::edge", "a.py::late", "b.py::f", "a.py::early"]
