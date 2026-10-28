import pytest

from pinpoint import tools


@pytest.fixture
def snap(tmp_path):
    """A tiny snapshot: one source file, one test file, and a secret file outside it."""
    root = tmp_path / "snap"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text("def alpha():\n    return 1\n\n\ndef beta():\n    return alpha()\n")
    (root / "tests").mkdir()
    (root / "tests" / "test_mod.py").write_text("def test_alpha():\n    alpha()\n")
    (tmp_path / "secret.txt").write_text("hidden\n")
    return root


def test_grep_finds_source_lines_and_skips_tests(snap):
    out = tools.grep(snap, r"alpha\(")
    assert out.splitlines() == ["pkg/mod.py:1: def alpha():", "pkg/mod.py:6: return alpha()"]


def test_grep_caps_matches(snap, monkeypatch):
    monkeypatch.setattr(tools, "MAX_MATCHES", 1)
    out = tools.grep(snap, "return")
    assert out.splitlines() == ["pkg/mod.py:2: return 1", "... 1 more matches not shown; use a more specific pattern"]


def test_grep_reports_bad_regex_and_no_matches(snap):
    assert tools.grep(snap, "(").startswith("error: invalid regex")
    assert tools.grep(snap, "zeta") == "no matches for 'zeta'"


def test_open_file_numbers_and_clamps(snap):
    out = tools.open_file(snap, "pkg/mod.py", 5, 99)
    assert out.splitlines() == ["pkg/mod.py lines 5-6 of 6", "    5| def beta():", "    6|     return alpha()"]


def test_open_file_caps_lines(snap, monkeypatch):
    monkeypatch.setattr(tools, "MAX_LINES", 2)
    assert tools.open_file(snap, "pkg/mod.py", 1, 6).splitlines()[0] == "pkg/mod.py lines 1-2 of 6"


@pytest.mark.parametrize("path", ["../secret.txt", "pkg/../../secret.txt"])
def test_open_file_stays_in_sandbox(snap, path):
    assert "outside the repository" in tools.open_file(snap, path, 1, 5)


def test_open_file_absolute_path_is_outside(snap, tmp_path):
    assert "outside the repository" in tools.open_file(snap, str(tmp_path / "secret.txt"), 1, 5)


def test_open_file_symlink_out_is_blocked(snap, tmp_path):
    (snap / "link.txt").symlink_to(tmp_path / "secret.txt")
    assert "outside the repository" in tools.open_file(snap, "link.txt", 1, 5)


def test_run_tool_dispatches_and_reports_errors(snap):
    assert tools.run_tool(snap, "grep", '{"pattern": "def beta"}') == "pkg/mod.py:5: def beta():"
    assert tools.run_tool(snap, "open_file", '{"path": "pkg/mod.py", "start": 1, "end": 1}').endswith("1| def alpha():")
    assert tools.run_tool(snap, "rm", "{}").startswith("error: unknown tool")
    assert tools.run_tool(snap, "grep", "{not json").startswith("error: bad arguments")
    assert tools.run_tool(snap, "open_file", '{"path": "pkg/mod.py"}').startswith("error: bad arguments")


@pytest.fixture
def nested(tmp_path):
    root = tmp_path / "snap"
    root.mkdir()
    (root / "m.py").write_text(
        "X = 1\n"                      # 1
        "class C:\n"                   # 2
        "    def meth(self):\n"        # 3
        "        def inner():\n"       # 4
        "            return 2\n"       # 5
        "        return inner()\n"     # 6
        "\n"                           # 7
        "def opts(p):\n"               # 8
        "    p.addoption('--flag')\n"  # 9
    )
    return root


def test_enclosing_is_innermost_function(nested):
    assert tools.enclosing(nested, "m.py", 5) == ("m.py::C.meth.inner", 4, 5)
    assert tools.enclosing(nested, "m.py", 6) == ("m.py::C.meth", 3, 6)
    assert tools.enclosing(nested, "m.py", 1) is None


def test_labelled_grep_and_open_file(nested):
    assert tools.grep(nested, "return 2", labels=True) == "m.py:5 [m.py::C.meth.inner]: return 2"
    assert tools.grep(nested, "^X", labels=True) == "m.py:1 [module level]: X = 1"
    out = tools.open_file(nested, "m.py", 3, 6, labels=True).splitlines()
    assert out[1:] == ["      ## m.py::C.meth (lines 3-6)", "    3|     def meth(self):",
                       "      ## m.py::C.meth.inner (lines 4-5)", "    4|         def inner():", "    5|             return 2",
                       "      ## m.py::C.meth (lines 3-6)", "    6|         return inner()"]
    assert "##" not in tools.open_file(nested, "m.py", 3, 6)  # labels off: unchanged output


def test_option_definitions(nested):
    assert tools.option_definitions(nested, "running with --flag and --missing fails") == ["--flag: m.py:9 [m.py::opts]"]


def test_call_graph_lists_callers_and_callees(snap):
    (snap / "pkg" / "other.py").write_text("def gamma():\n    beta()\n    print(1)\n")
    out = tools.call_graph(snap, "pkg/mod.py::beta").splitlines()
    assert out == ["pkg/mod.py::beta (lines 5-6)",
                   "called by 1 functions (calls to 'beta', matched by name):",
                   "  pkg/other.py::gamma (lines 1-3)",
                   "calls:",
                   "  alpha -> pkg/mod.py::alpha (lines 1-2)"]
    assert "1 more names not defined" in tools.call_graph(snap, "pkg/other.py::gamma")   # print
    assert "test_alpha" not in tools.call_graph(snap, "pkg/mod.py::alpha")               # tests are excluded


def test_call_graph_flags_ambiguous_names_and_unknown_ids(snap):
    (snap / "pkg" / "other.py").write_text("def alpha():\n    pass\n")
    out = tools.call_graph(snap, "pkg/mod.py::alpha")
    assert "2 functions are named 'alpha'" in out
    assert tools.call_graph(snap, "pkg/mod.py::zeta").startswith("error: no function")
    assert tools.run_tool(snap, "call_graph", '{"function_id": "pkg/mod.py::beta"}').startswith("pkg/mod.py::beta")
