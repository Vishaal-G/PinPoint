"""Collect 👍/👎 reactions on the bot's comments as feedback.

GitHub Actions has no event for reactions, so this *pulls*: a scheduled workflow
(.github/workflows/pinpoint-feedback.yml) lists the repo's issue comments, keeps the ones the bot
wrote (MARKER, bot author), and reads each one's reactions plus the hidden pinpoint-meta record
(what we suggested). Every run is a full snapshot, not an append log: people add and remove
reactions at any time, so the latest snapshot is the truth.

Explicit feedback is sparse and self-selected. Always read approval next to coverage:
    coverage = comments with >= 1 human vote / comments
    approval = 👍 / (👍 + 👎)            over those votes

    uv run python -m pinpoint.feedback --repo Vishaal-G/PinPoint --out data/gold/feedback.csv
"""
import argparse
import csv
import os
import sys
from datetime import datetime, timezone

from pinpoint.bot import MARKER, read_meta

FIELDS = ["comment_url", "issue", "issue_state", "created_at", "updated_at", "commit", "source", "top_suspect",
          "n_suspects", "up", "down", "voters", "author_voted", "collected_at"]


def rows(gh_repo, now: datetime | None = None) -> list[dict]:
    """One row per bot comment. `gh_repo` is a PyGithub Repository (or a fake with the same shape)."""
    now = now or datetime.now(timezone.utc)
    out = []
    for c in gh_repo.get_issues_comments():          # every issue comment in the repo, one paged listing
        if MARKER not in (c.body or "") or c.user.type != "Bot":
            continue
        meta = read_meta(c.body) or {}               # comments from before the meta line: reactions only
        up, down = set(), set()
        for r in c.get_reactions():
            if r.user.type == "Bot":                 # bots don't count as feedback
                continue
            {"+1": up, "-1": down}.get(r.content, set()).add(r.user.login)
        issue_no = int(c.issue_url.rsplit("/", 1)[1])
        issue = gh_repo.get_issue(issue_no)
        suspects = meta.get("suspects", [])
        out.append({
            "comment_url": c.html_url, "issue": issue_no, "issue_state": issue.state,
            "created_at": c.created_at.isoformat(), "updated_at": c.updated_at.isoformat(),
            "commit": meta.get("commit", ""), "source": meta.get("source", ""),
            "top_suspect": suspects[0] if suspects else "", "n_suspects": len(suspects),
            "up": len(up), "down": len(down), "voters": " ".join(sorted(up | down)),
            # The issue author knows the bug best: their vote is the most informed one.
            "author_voted": issue.user.login in (up | down),
            "collected_at": now.isoformat(timespec="seconds"),
        })
    return out


def summarize(rs: list[dict]) -> dict:
    rated = [r for r in rs if r["up"] + r["down"]]
    up, down = sum(r["up"] for r in rs), sum(r["down"] for r in rs)
    return {"comments": len(rs), "rated": len(rated), "coverage": len(rated) / len(rs) if rs else 0.0,
            "up": up, "down": down, "approval": up / (up + down) if up + down else None}


def markdown(s: dict, rs: list[dict]) -> str:
    approval = f"{s['approval']:.0%}" if s["approval"] is not None else "n/a"
    lines = ["### Pinpoint feedback", "",
             f"{s['comments']} comments, {s['rated']} rated (coverage {s['coverage']:.0%}); "
             f"👍 {s['up']} / 👎 {s['down']} (approval {approval})", ""]
    if s["rated"] < 20:
        lines += ["_Fewer than 20 rated comments: too few votes for the approval rate to mean much._", ""]
    lines += ["| Issue | Source | Top suspect | 👍 | 👎 |", "|---|---|---|---|---|"]
    lines += [f"| [#{r['issue']}]({r['comment_url']}) | {r['source'] or '?'} | `{r['top_suspect'] or '?'}` "
              f"| {r['up']} | {r['down']} |" for r in rs]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from github import Auth, Github

    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"), help="owner/name")
    p.add_argument("--out", default="feedback.csv")
    args = p.parse_args(argv)
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    gh = Github(auth=Auth.Token(token)) if token else Github()

    rs = rows(gh.get_repo(args.repo))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rs)
    report = markdown(summarize(rs), rs)
    print(report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):         # shows on the Actions run page
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(report + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
