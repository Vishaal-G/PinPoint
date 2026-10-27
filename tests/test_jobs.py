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
