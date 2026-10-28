import pandas as pd

from pinpoint.split import heldout_counts, make_split, tune_ids


def fake_issues(sizes: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame([{"instance_id": f"{repo}-{i}", "repo": repo}
                         for repo, n in sizes.items() for i in range(n)])


SIZES = {"big": 114, "mid": 77, "small": 6, "tiny": 3}


def test_counts_are_proportional_and_exact():
    counts = heldout_counts({"a": 9, "b": 6, "c": 3}, 6)
    assert counts == {"a": 3, "b": 2, "c": 1}
    counts = heldout_counts(SIZES, 67)
    assert sum(counts.values()) == 67
    for repo, n in SIZES.items():
        assert abs(counts[repo] - n * 67 / 200) < 1  # never more than one off its exact share


def test_split_sizes_and_disjoint():
    split = make_split(fake_issues(SIZES), n_heldout=67)
    assert len(split) == 200 and split.instance_id.is_unique
    assert (split.split == "heldout").sum() == 67
    assert (split.split == "tune").sum() == 133


def test_split_is_stratified():
    split = make_split(fake_issues(SIZES), n_heldout=67)
    per_repo = split[split.split == "heldout"].groupby("repo").size().to_dict()
    assert per_repo == heldout_counts(SIZES, 67)


def test_same_seed_same_split_regardless_of_row_order():
    issues = fake_issues(SIZES)
    a = make_split(issues, n_heldout=67, seed=1)
    b = make_split(issues.sample(frac=1, random_state=5), n_heldout=67, seed=1)
    pd.testing.assert_frame_equal(a, b)
    c = make_split(issues, n_heldout=67, seed=2)
    assert not a.split.equals(c.split)


def test_tune_ids_reads_saved_file(tmp_path):
    path = tmp_path / "split.csv"
    make_split(fake_issues({"r": 6}), n_heldout=2).to_csv(path, index=False)
    assert len(tune_ids(path)) == 4


def test_recheck_ids_is_a_stable_stratified_tune_slice():
    from pinpoint.split import load_split, recheck_ids, tune_ids

    ids = recheck_ids()
    assert ids == recheck_ids() and ids <= tune_ids()
    per_repo = load_split().set_index("instance_id").loc[sorted(ids)].repo.value_counts()
    assert per_repo["pytest-dev/pytest"] == 11
    assert per_repo.drop("pytest-dev/pytest").max() == 4 and len(per_repo) == 12
