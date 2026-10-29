from pinpoint.jobs import in_list, parse_args, table_name


def test_defaults_point_at_the_notebook_tables():
    args = parse_args([])
    assert table_name(args, "silver", "functions") == "workspace.pinpoint_silver.functions"
    assert args.repo == "pytest-dev/pytest"


def test_job_parameters_override_defaults():
    args = parse_args(["--catalog", "main", "--schema-prefix", "pp_dev", "--repo", "django/django"])
    assert table_name(args, "gold", "ground_truth") == "main.pp_dev_gold.ground_truth"
    assert args.repo == "django/django"


def test_in_list_is_sorted_and_unique():
    assert in_list(["b-2", "a-1", "b-2"]) == "'a-1', 'b-2'"


def test_plain_dtypes_turns_arrow_strings_into_objects_and_keeps_numbers():
    import pandas as pd

    from pinpoint.jobs import plain_dtypes

    pdf = pd.concat([pd.DataFrame({"s": pd.array(["a"], dtype="string[pyarrow]"), "n": [1], "b": [True]})] * 2)
    out = plain_dtypes(pdf)
    assert out["s"].dtype == object and out["n"].dtype == "int64" and out["b"].dtype == bool
    assert out["s"].tolist() == ["a", "a"]


def test_slices_select_issues_and_never_heldout():
    import pandas as pd

    from pinpoint.jobs import embeddings_table, select_issues, slice_label
    from pinpoint.split import load_split, recheck_ids, tune_ids

    split = load_split()
    issues = split[["instance_id", "repo"]]
    assert set(select_issues(issues, parse_args(["--slice", "recheck"])).instance_id) == recheck_ids()
    assert set(select_issues(issues, parse_args(["--slice", "tune"])).instance_id) == tune_ids()
    flask = select_issues(issues, parse_args(["--repo", "pallets/flask"]))
    assert set(flask.repo) == {"pallets/flask"} and len(flask) == (split.repo == "pallets/flask").sum()
    assert slice_label(parse_args(["--slice", "recheck"])) == "recheck"
    assert embeddings_table(parse_args([])) == "workspace.pinpoint_silver.embeddings_databricks_bge_large_en"
    assert isinstance(issues, pd.DataFrame)


def test_eval_chunks_are_fixed_size_and_one_repo_each():
    import pandas as pd

    from pinpoint.jobs import eval_chunks

    issues = pd.DataFrame({"instance_id": [f"b-{i}" for i in range(5)] + ["a-1", "a-0"],
                           "repo": ["b"] * 5 + ["a"] * 2})
    chunks = eval_chunks(issues, size=2)
    assert [c.instance_id.tolist() for c in chunks] == [["a-0", "a-1"], ["b-0", "b-1"], ["b-2", "b-3"], ["b-4"]]
    assert eval_chunks(issues.sample(frac=1, random_state=0), size=2)[2].instance_id.iloc[0] == "b-2"  # order-proof


def test_chunked_rankings_equal_per_issue_rankings():
    """The job ranks a chunk of issues from one merged frame; that must give each issue the same
    ranking as the local path (candidates_from on that issue alone)."""
    import numpy as np
    import pandas as pd

    from pinpoint import evaluate as ev
    from pinpoint.jobs import chunk_candidates
    from pinpoint.repos import is_source_file
    from test_evaluate import fixture

    membership, functions, vectors, edges, _, _ = fixture()
    other = membership.assign(instance_id="j", start_line=membership.start_line[::-1].values)  # same bodies
    members = pd.concat([other, membership])
    members = members[members.path.map(is_source_file)]
    issues = pd.DataFrame({"instance_id": ["i", "j"], "problem_statement": ["fix crashes", "caller"]})
    queries = {"i": np.array([0, 0.6, 0.8]), "j": np.array([0.6, 0.8, 0])}
    chunked = ev.rankings(issues, chunk_candidates(members, functions), vectors, queries, edges)
    for iid in ("i", "j"):
        alone = ev.rankings(issues[issues.instance_id == iid],
                            ev.candidates_from(members[members.instance_id == iid], functions), vectors, queries, edges)
        assert {name: r[iid] for name, r in chunked.items()} == {name: r[iid] for name, r in alone.items()}


def test_checkpoints_round_trip_full_rankings_and_ignore_partial_files(tmp_path):
    from pinpoint.evaluate import STRATEGIES
    from pinpoint.jobs import load_checkpoints, save_checkpoint

    long = [f"f{i}" for i in range(80)] + ["f3"]  # longer than eval_rankings' top-50, with a repeat
    ranked = {name: {"i": long, "j": ["x"]} for name in STRATEGIES}
    save_checkpoint(tmp_path, "i", ranked)
    (tmp_path / "k.parquet.part").write_bytes(b"killed mid-write")
    done, loaded = load_checkpoints(tmp_path)
    assert done == {"i"} and loaded == ranked
    assert load_checkpoints(tmp_path / "missing") == (set(), {name: {} for name in STRATEGIES})
