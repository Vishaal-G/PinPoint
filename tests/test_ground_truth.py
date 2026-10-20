from pinpoint.ground_truth import function_id, gold_functions
from pinpoint.parse import parse_snapshot

SAMPLE = '''\
import os
import sys

LIMIT = 10


def top(x):
    return x


class Greeter:
    def greet(self, name):
        return name

    def later(self):
        def helper():
            return 1
        return helper()
'''
# Lines: 1-2 imports, 4 constant, 7-8 top, 11 class, 12-13 greet, 15-18 later, 16-17 helper


def functions(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(SAMPLE)
    return parse_snapshot(tmp_path)


def test_function_id():
    assert function_id("src/a.py", "Cls.meth") == "src/a.py::Cls.meth"


def test_line_inside_a_function(tmp_path):
    assert gold_functions({"pkg/mod.py": {8}}, functions(tmp_path)) == ["pkg/mod.py::top"]


def test_method_gets_class_qualified_name(tmp_path):
    assert gold_functions({"pkg/mod.py": {13}}, functions(tmp_path)) == ["pkg/mod.py::Greeter.greet"]


def test_nested_line_goes_to_innermost_function(tmp_path):
    assert gold_functions({"pkg/mod.py": {17}}, functions(tmp_path)) == [
        "pkg/mod.py::Greeter.later.helper"
    ]


def test_parent_line_outside_nested_goes_to_parent(tmp_path):
    assert gold_functions({"pkg/mod.py": {18}}, functions(tmp_path)) == ["pkg/mod.py::Greeter.later"]


def test_lines_outside_functions_are_ignored(tmp_path):
    # imports, a module constant, a blank line between defs, the class line itself
    assert gold_functions({"pkg/mod.py": {1, 4, 9, 11}}, functions(tmp_path)) == []


def test_several_lines_several_functions_deduplicated(tmp_path):
    changed = {"pkg/mod.py": {2, 7, 8, 12, 13}}
    assert gold_functions(changed, functions(tmp_path)) == [
        "pkg/mod.py::Greeter.greet",
        "pkg/mod.py::top",
    ]


def test_file_not_in_snapshot_gives_nothing(tmp_path):
    assert gold_functions({"other.py": {1}}, functions(tmp_path)) == []
