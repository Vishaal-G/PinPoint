"""Lock the tune / held-out split. Tune is for every experiment; held-out is scored once, in Week 9.

    uv run python -m pinpoint.split        # writes src/pinpoint/split.csv (refuses to overwrite)

The split is stratified by repo: each repo gives about a third of its issues to held-out, so
both parts have the same mix (django-heavy, a few flask issues) and neither is easier by luck.
The file is committed to git inside the package (not under data/), so the split can't silently change
and the Databricks job wheel carries it.
"""
import random
from pathlib import Path

import pandas as pd

from pinpoint.data import load_issues

SPLIT_PATH = Path(__file__).with_name("split.csv")  # inside the package, so the job wheel ships it
SEED = 20261022
N_HELDOUT = 100


def heldout_counts(repo_sizes: dict[str, int], n_heldout: int) -> dict[str, int]:
    """How many held-out issues each repo gives, proportional to its size, summing to n_heldout.
    Largest remainder: floor every repo's exact share, then hand the leftover slots to the
    repos whose share was cut the most."""
    total = sum(repo_sizes.values())
    exact = {r: n * n_heldout / total for r, n in repo_sizes.items()}
    counts = {r: int(x) for r, x in exact.items()}
    leftover = n_heldout - sum(counts.values())
    # Ties broken by repo name, so the result never depends on dict order.
    for r in sorted(exact, key=lambda r: (counts[r] - exact[r], r))[:leftover]:
        counts[r] += 1
    return counts


def make_split(issues: pd.DataFrame, n_heldout: int = N_HELDOUT, seed: int = SEED) -> pd.DataFrame:
    """One row per issue: instance_id, repo, split ("tune" or "heldout")."""
    rng = random.Random(seed)
    sizes = issues.groupby("repo").size().to_dict()
    counts = heldout_counts(sizes, n_heldout)
    rows = []
    for repo in sorted(sizes):
        # Sort before shuffling, so the split depends only on the seed, not on row order.
        ids = sorted(issues.loc[issues.repo == repo, "instance_id"])
        rng.shuffle(ids)
        rows += [{"instance_id": iid, "repo": repo,
                  "split": "heldout" if i < counts[repo] else "tune"}
                 for i, iid in enumerate(ids)]
    return pd.DataFrame(rows).sort_values("instance_id").reset_index(drop=True)


def load_split(path: Path = SPLIT_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def tune_ids(path: Path = SPLIT_PATH) -> set[str]:
    split = load_split(path)
    return set(split.loc[split.split == "tune", "instance_id"])


RECHECK_PER_REPO = 4
RECHECK_KEEP_ALL = ("pytest-dev/pytest",)  # the development slice: keep all, so old rows compare


def recheck_ids(path: Path = SPLIT_PATH, per_repo: int = RECHECK_PER_REPO, seed: int = SEED) -> set[str]:
    """A small stratified slice of tune for checking the pipeline across every repo before all 200:
    up to `per_repo` tune issues from each repo (all of pytest). Same ids every call."""
    rng = random.Random(seed)
    tune = load_split(path)
    tune = tune[tune.split == "tune"]
    out = set()
    for repo in sorted(tune.repo.unique()):
        ids = sorted(tune.loc[tune.repo == repo, "instance_id"])
        rng.shuffle(ids)
        out |= set(ids if repo in RECHECK_KEEP_ALL else ids[:per_repo])
    return out


if __name__ == "__main__":
    if SPLIT_PATH.exists():
        raise SystemExit(f"{SPLIT_PATH} already exists. The split is locked; delete it by hand only if you mean it.")
    split = make_split(load_issues())
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(SPLIT_PATH, index=False)
    print(split.groupby(["repo", "split"]).size().unstack(fill_value=0).to_string())
    print(split.split.value_counts().to_string())
