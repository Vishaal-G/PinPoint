import mlflow
import pytest

from pinpoint import tracking


@pytest.fixture
def tmp_tracking(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    return tracking.TRACKING_URI


def test_per_issue_table():
    df = tracking.per_issue(
        {"A": ["x", "run"], "B": ["y"]},
        {"A": ["run"], "B": ["z"], "C": []},
    )
    assert df["instance_id"].tolist() == ["A", "B"]  # C has no gold, skipped
    assert df["gold_rank"].tolist()[0] == 2
    assert df["gold_rank"].isna().tolist() == [False, True]
    assert df["top1"].tolist() == ["x", "y"]


def test_log_run_records_params_and_metrics(tmp_tracking):
    results = tracking.log_run(
        "unit-test", {"strategy": "fake", "k1": 1.5},
        rankings={"A": ["x", "run"]}, gold={"A": ["run"]}, ks=(1, 5),
    )
    assert results["acc@5"] == 1.0

    runs = mlflow.search_runs(experiment_names=[tracking.EXPERIMENT])
    run = runs.iloc[0]
    assert run["tags.mlflow.runName"] == "unit-test"
    assert run["params.strategy"] == "fake"
    assert run["params.n"] == "1"
    assert run["metrics.acc_at_1"] == 0.0
    assert run["metrics.acc_at_5"] == 1.0
    assert run["metrics.mrr"] == 0.5
