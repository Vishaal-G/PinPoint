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
