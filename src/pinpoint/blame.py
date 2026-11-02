"""Who to ask about a function: the author of most of its current lines, from `git blame`.

    uv run python -m pinpoint.blame <checkout> <path> <start> <end>
"""
import sys
from collections import Counter
from pathlib import Path


def owner(checkout: Path, path: str, start: int, end: int, rev: str = "HEAD") -> dict | None:
    """The author with the most lines in `path` lines start..end at `rev`: name, email, lines, share.
    Ties go to the most recent author (freshest context). Bots are skipped. None if the checkout
    isn't a git repo or blame fails (e.g. path not at rev)."""
    from git import GitCommandError, InvalidGitRepositoryError, NoSuchPathError, Repo

    try:
        repo = Repo(checkout, search_parent_directories=True)
        # -L limits blame to the function's lines; -w ignores whitespace-only changes (a reformat
        # commit would otherwise "own" everything it re-indented).
        hunks = repo.blame(rev, path, L=f"{start},{end}", w=True)
    except (GitCommandError, InvalidGitRepositoryError, NoSuchPathError, ValueError):
        return None
    lines, who, latest = Counter(), {}, {}
    for commit, hunk_lines in hunks:
        email = commit.author.email
        if "[bot]" in (commit.author.name or "") or email.endswith("[bot]@users.noreply.github.com"):
            continue
        lines[email] += len(hunk_lines)
        who[email] = commit.author.name
        latest[email] = max(latest.get(email, 0), commit.authored_date)
    if not lines:
        return None
    email = max(lines, key=lambda e: (lines[e], latest[e]))
    n = lines[email]
    return {"name": who[email], "email": email, "lines": n, "share": round(n / sum(lines.values()), 2)}


if __name__ == "__main__":
    print(owner(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])))
