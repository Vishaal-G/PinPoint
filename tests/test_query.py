from pinpoint.query import extract, looks_like_code, rewrite, title

ISSUE = """Rewrite fails when first expression of file is a number
Running `pytest --fixtures` on a module crashes in `AssertionRewriter.run()`.
```
$ pip list
attrs 22.1.0
pluggy 1.0.0
/usr/lib/python3.8/site-packages/_pytest/assertion/rewrite.py:760: in is_rewrite_disabled
    return "PYTEST_DONT_REWRITE" in docstring
E   TypeError: argument of type 'int' is not iterable
```
Traceback:
  File "src/_pytest/python.py", line 608, in _importtestmodule
"""


def test_frames_from_both_traceback_styles():
    assert extract(ISSUE).frames == ["is_rewrite_disabled", "_importtestmodule"]


def test_error_names():
    assert extract(ISSUE).errors == ["TypeError"]


def test_identifiers_keep_code_drop_words_and_versions():
    ids = extract(ISSUE).identifiers
    assert "--fixtures" in ids and "AssertionRewriter.run" in ids and "PYTEST_DONT_REWRITE" in ids
    assert "attrs" not in ids and "pluggy" not in ids and "pip" not in ids


def test_paths_trimmed_to_the_package():
    assert extract(ISSUE).paths == ["_pytest/assertion/rewrite.py", "src/_pytest/python.py"]


def test_looks_like_code():
    assert all(map(looks_like_code, ["is_x", "os.path", "getBasetemp", "--fixtures", "run()"]))
    assert not any(map(looks_like_code, ["attrs", "the", "Version"]))


def test_rewrite_modes():
    assert rewrite(ISSUE, "raw") == ISSUE
    assert rewrite(ISSUE, "terms").startswith("is_rewrite_disabled _importtestmodule TypeError")
    assert rewrite(ISSUE, "title+terms").startswith(title(ISSUE) + "\n")
    assert rewrite("nothing code-like here", "terms") == "nothing code-like here"   # falls back to raw


def test_separator_lines_are_not_code():
    assert not looks_like_code("______") and not looks_like_code("--------")
    assert extract("```\n____ test ____\n---------\n```").identifiers == []


def test_repo_vocab_drops_the_reporters_own_names():
    from pinpoint.query import in_vocab

    vocab = {"assertionrewriter", "run", "typeerror"}
    assert in_vocab(["AssertionRewriter.run", "test_all_even", "TypeError"], vocab) == ["AssertionRewriter.run", "TypeError"]
    assert rewrite(ISSUE, "repo-terms", vocab) == "TypeError AssertionRewriter.run"
