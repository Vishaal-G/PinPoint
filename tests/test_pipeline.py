import pandas as pd

from pinpoint.pipeline import merge_tables


def tables(fn_hashes, members):
    return {"functions": pd.DataFrame({"hash": fn_hashes, "source": [f"src {h}" for h in fn_hashes]}),
            "edges": pd.DataFrame({"hash": fn_hashes, "callee": ["f"] * len(fn_hashes)}),
            "snapshot_membership": pd.DataFrame({"instance_id": [i for i, _ in members], "hash": [h for _, h in members]}),
            "ground_truth": pd.DataFrame({"instance_id": sorted({i for i, _ in members})})}


def test_merge_keeps_one_row_per_hash_and_replaces_rebuilt_issues():
    old = tables(["a", "b"], [("i1", "a"), ("i2", "b")])
    new = tables(["b", "c"], [("i2", "c")])           # i2 rebuilt, now contains c instead of b
    out = merge_tables(old, new)
    assert sorted(out["functions"].hash) == ["a", "b", "c"]
    assert sorted(map(tuple, out["snapshot_membership"].values.tolist())) == [("i1", "a"), ("i2", "c")]
    assert sorted(out["ground_truth"].instance_id) == ["i1", "i2"]
    again = merge_tables(out, new)                      # idempotent
    assert all(again[k].sort_values(list(again[k].columns)).reset_index(drop=True).equals(
        out[k].sort_values(list(out[k].columns)).reset_index(drop=True)) for k in out)


def test_merge_into_empty():
    new = tables(["a"], [("i1", "a")])
    assert merge_tables({}, new)["functions"].equals(new["functions"])
