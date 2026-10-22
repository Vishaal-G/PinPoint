from pinpoint.bm25 import rank_files


def write(root, files):
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)


def test_ranks_matching_file_first(tmp_path):
    write(tmp_path, {
        "pkg/rewrite.py": "def is_rewrite_disabled(docstring):\n    return 'rewrite' in docstring\n",
        "pkg/config.py": "def load_config(path):\n    return open(path).read()\n",
        "pkg/terminal.py": "def write_line(text):\n    print(text)\n",
    })
    ranking = rank_files(tmp_path, "Rewrite fails when docstring is a number")
    assert ranking[0] == "pkg/rewrite.py"
    assert sorted(ranking) == ["pkg/config.py", "pkg/rewrite.py", "pkg/terminal.py"]


def test_test_files_are_not_candidates(tmp_path):
    write(tmp_path, {
        "pkg/mod.py": "def f():\n    pass\n",
        "testing/test_mod.py": "def test_docstring_rewrite():\n    pass\n",
    })
    assert rank_files(tmp_path, "docstring rewrite") == ["pkg/mod.py"]


def test_split_mode_matches_words_inside_identifiers(tmp_path):
    write(tmp_path, {
        "a.py": "def is_rewrite_disabled():\n    pass\n",
        "b.py": "def load():\n    pass\n",
        "c.py": "def save():\n    pass\n",
    })
    assert rank_files(tmp_path, "rewrite disabled", mode="split")[0] == "a.py"


def test_run_scores_only_the_issues_it_ranked(tmp_path, monkeypatch):
    import pandas as pd
    from pinpoint import bm25

    write(tmp_path, {"pkg/rewrite.py": "def rewrite():\n    pass\n", "pkg/other.py": "x = 1\n"})
    monkeypatch.setattr(bm25, "get_snapshot", lambda issue: tmp_path)
    monkeypatch.setattr(bm25, "log_run", lambda name, params, rankings, gold: gold)
    issues = pd.DataFrame([{"instance_id": "A", "problem_statement": "rewrite"}])
    # "B" is in gold (e.g. a held-out issue) but not in issues: it must not count as a miss.
    gold = bm25.run(issues, {"A": ["pkg/rewrite.py"], "B": ["pkg/other.py"]})
    assert list(gold) == ["A"]
