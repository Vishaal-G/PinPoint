import numpy as np
import pandas as pd

from pinpoint.dense import rank_files
from pinpoint.dense_functions import chunk_texts, rank_functions

FNS = pd.DataFrame({
    "path":        ["a.py", "a.py", "b.py"],
    "qualname":    ["f", "Cls.g", "h"],
    "function_id": ["a.py::f", "a.py::Cls.g", "b.py::h"],
    "source":      ["def f(): pass", "def g(self): pass", "def h(): pass"],
})


def test_each_function_is_ranked_by_its_own_score():
    assert rank_functions(FNS, np.array([0.1, 0.7, 0.4])) == ["a.py::Cls.g", "b.py::h", "a.py::f"]


def test_file_is_ranked_by_its_best_function():
    assert rank_files(FNS, np.array([0.1, 0.3, 0.4])) == ["b.py", "a.py"]


def test_header_adds_path_and_qualname():
    assert chunk_texts(FNS)[1] == "def g(self): pass"
    assert chunk_texts(FNS, header=True)[1] == "a.py\nCls.g\ndef g(self): pass"
