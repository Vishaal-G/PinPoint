import shutil

import pytest
from git import Actor, Repo

from pinpoint import repos

AUTHOR = Actor("test", "test@example.com")


@pytest.fixture
def fake_clone(tmp_path, monkeypatch):
    """A tiny local repo with two commits, placed where get_repo expects the clone,
    so tests never touch GitHub. Returns (repo, first_sha, second_sha)."""
    monkeypatch.setattr(repos, "REPOS_DIR", tmp_path / "repos")
    monkeypatch.setattr(repos, "WORKTREES_DIR", tmp_path / "worktrees")

    repo = Repo.init(tmp_path / "repos" / "proj")
    f = tmp_path / "repos" / "proj" / "mod.py"
    f.write_text("x = 1\n")
    repo.index.add(["mod.py"])
    first = repo.index.commit("first", author=AUTHOR, committer=AUTHOR).hexsha
    f.write_text("x = 2\n")
    repo.index.add(["mod.py"])
    second = repo.index.commit("second", author=AUTHOR, committer=AUTHOR).hexsha
    return repo, first, second


def issue(instance_id, commit):
    return {"repo": "owner/proj", "instance_id": instance_id, "base_commit": commit}


def test_snapshot_is_at_base_commit(fake_clone):
    _, first, _ = fake_clone
    wt = repos.get_snapshot(issue("a", first))
    assert Repo(wt).head.commit.hexsha == first
    assert (wt / "mod.py").read_text() == "x = 1\n"


def test_two_issues_get_separate_snapshots(fake_clone):
    _, first, second = fake_clone
    wt1 = repos.get_snapshot(issue("a", first))
    wt2 = repos.get_snapshot(issue("b", second))
    assert wt1 != wt2
    assert (wt1 / "mod.py").read_text() == "x = 1\n"
    assert (wt2 / "mod.py").read_text() == "x = 2\n"


def test_get_snapshot_twice_reuses_it(fake_clone):
    _, first, _ = fake_clone
    assert repos.get_snapshot(issue("a", first)) == repos.get_snapshot(issue("a", first))


def test_snapshot_path_is_absolute(fake_clone):
    _, first, _ = fake_clone
    assert repos.get_snapshot(issue("a", first)).is_absolute()


def test_remove_snapshot(fake_clone):
    repo, first, _ = fake_clone
    wt = repos.get_snapshot(issue("a", first))
    repos.remove_snapshot(issue("a", first))
    assert not wt.exists()
    assert str(wt) not in repo.git.worktree("list")


def test_remove_missing_snapshot_is_a_no_op(fake_clone):
    _, first, _ = fake_clone
    repos.remove_snapshot(issue("never-made", first))


def test_recreate_after_folder_deleted_by_hand(fake_clone):
    _, first, _ = fake_clone
    wt = repos.get_snapshot(issue("a", first))
    shutil.rmtree(wt)
    wt = repos.get_snapshot(issue("a", first))
    assert Repo(wt).head.commit.hexsha == first


def test_is_source_file():
    assert repos.is_source_file("src/_pytest/assertion/rewrite.py")
    assert repos.is_source_file("setup.py")
    assert not repos.is_source_file("testing/test_assertrewrite.py")
    assert not repos.is_source_file("tests/helpers.py")
    assert not repos.is_source_file("src/pkg/test_utils.py")
    assert not repos.is_source_file("src/pkg/utils_test.py")
    assert not repos.is_source_file("conftest.py")
    assert not repos.is_source_file("doc/en/conf.py")
    assert not repos.is_source_file("README.md")
