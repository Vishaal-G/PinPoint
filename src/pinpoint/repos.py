from pathlib import Path

from git import Repo

REPOS_DIR = Path("data/repos")
WORKTREES_DIR = Path("data/worktrees")


def _clone_path(repo_name: str) -> Path:
    """Where the clone of a repo lives, e.g. "pytest-dev/pytest" -> data/repos/pytest."""
    return REPOS_DIR / repo_name.split("/")[1]


def _worktree_path(issue) -> Path:
    """Where one issue's snapshot lives, e.g. data/worktrees/pytest-dev__pytest-11143."""
    return WORKTREES_DIR / issue["instance_id"]


def get_repo(repo_name: str) -> Repo:
    """Open the clone of a repo, cloning it from GitHub first if it isn't there."""
    path = _clone_path(repo_name)
    if path.exists():
        return Repo(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return Repo.clone_from(f"https://github.com/{repo_name}.git", path)


def get_snapshot(issue) -> Path:
    """Return a worktree checked out at the issue's base_commit, creating it if missing."""
    wt_path = _worktree_path(issue).resolve()
    if wt_path.exists():
        return wt_path

    repo = get_repo(issue["repo"])
    WORKTREES_DIR.mkdir(parents=True, exist_ok=True)
    # Forget worktrees whose folders were deleted by hand, or `add` refuses the path.
    repo.git.worktree("prune")
    # git runs inside the clone, so the path must be absolute.
    repo.git.worktree("add", "--detach", str(wt_path), issue["base_commit"])
    return wt_path


def remove_snapshot(issue) -> None:
    """Delete the issue's worktree, if there is one."""
    wt_path = _worktree_path(issue).resolve()
    if not wt_path.exists():
        return
    repo = get_repo(issue["repo"])
    repo.git.worktree("remove", "--force", str(wt_path))
