"""The GitHub bot: runs inside a GitHub Actions job on `issues: opened` and comments where to look.

The workflow (.github/workflows/pinpoint.yml in the target repo) checks the repo out with full
history (blame needs it), installs `pinpoint[bot]`, and runs `pinpoint-bot`. Everything comes from
the environment Actions provides:

    GITHUB_EVENT_PATH   the webhook payload (the issue)        GITHUB_REPOSITORY  owner/name
    GITHUB_WORKSPACE    the checkout = the snapshot             GITHUB_TOKEN       posts the comment
    DATABRICKS_HOST / DATABRICKS_TOKEN   embeddings + LLM       PINPOINT_CACHE     embedding cache dir

Nothing listens on the internet: no inbound auth to get wrong. Issue text is untrusted data; it
never reaches a shell, and the agent's tools are read-only inside the checkout.

    uv run python -m pinpoint.bot --event event.json --repo o/r --workspace path --dry-run   # local test
"""
import argparse
import json
import os
import sys
from pathlib import Path

from pinpoint import live
from pinpoint.explain import CITATION

MARKER = "<!-- pinpoint -->"  # finds our earlier comment on a rerun, so we edit instead of piling up


def link(repo: str, sha: str, path: str, start: int, end: int | None = None) -> str:
    return f"https://github.com/{repo}/blob/{sha}/{path}#L{start}" + (f"-L{end}" if end and end != start else "")


def defang(text: str) -> str:
    """Model output goes into a public comment: break @mentions (no pinging people from a bot)."""
    return text.replace("@", "@​")


def format_comment(rep: dict) -> str:
    repo, sha = rep["repo"], rep["commit"]
    rows = []
    for i, s in enumerate(rep["suspects"], 1):
        qualname = s["function_id"].split("::", 1)[1]
        o = s["owner"]
        who = f"{o['name']} ({o['share']:.0%} of lines)" if o else "unknown"
        rows.append(f"| {i} | [`{qualname}`]({link(repo, sha, s['path'], s['start_line'], s['end_line'])}) "
                    f"| `{s['path']}` | {s['start_line']}-{s['end_line']} | {defang(who)} |")
    out = [MARKER, "### Pinpoint: where to look", "",
           "| # | Function | File | Lines | Most lines by |", "|---|---|---|---|---|", *rows]
    if rep["explanation"]:
        valid = {(c["path"], c["line"]) for c in rep["explanation"]["citations"] if c["valid"]}
        text = CITATION.sub(lambda m: f"[{m[0]}]({link(repo, sha, m[1], int(m[2]))})"
                            if (m[1], int(m[2])) in valid else m[0], defang(rep["explanation"]["text"]))
        out += ["", f"**Why:** {text}"]
    source = "agent (dense retrieval + Llama 3.3 70B)" if rep["source"] == "agent" else "dense retrieval only"
    out += ["", f"<sub>Ranked by {source} at {sha[:7]}. React with 👍 or 👎 to rate this comment.</sub>"]
    if rep["warnings"]:
        out += ["", "<details><summary>Warnings</summary>", "", *(f"- {defang(w)}" for w in rep["warnings"]),
                "", "</details>"]
    return "\n".join(out)


def post(repo: str, number: int, body: str, token: str) -> str:
    """Create our comment, or edit the one we left before. Returns its URL."""
    from github import Auth, Github

    issue = Github(auth=Auth.Token(token)).get_repo(repo).get_issue(number)
    for c in issue.get_comments():
        if MARKER in (c.body or "") and c.user.type == "Bot":
            c.edit(body)
            return c.html_url
    return issue.create_comment(body).html_url


def main(argv: list[str] | None = None) -> int:
    from pinpoint.agent import databricks_llm
    from pinpoint.embed import ENDPOINT_MODEL, EndpointEmbedder

    p = argparse.ArgumentParser()
    p.add_argument("--event", default=os.environ.get("GITHUB_EVENT_PATH"), help="webhook payload JSON")
    p.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"), help="owner/name")
    p.add_argument("--workspace", default=os.environ.get("GITHUB_WORKSPACE", "."), help="the checkout")
    p.add_argument("--cache", default=os.environ.get("PINPOINT_CACHE", str(Path.home() / ".cache" / "pinpoint")))
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--dry-run", action="store_true", help="print the comment instead of posting it")
    args = p.parse_args(argv)

    issue = json.loads(Path(args.event).read_text())["issue"]
    if issue.get("user", {}).get("type") == "Bot":
        print("issue opened by a bot: skipping")
        return 0
    snapshot, sha = live.checkout(args.workspace)  # a local directory: used as is
    text = f"{issue['title']}\n{issue.get('body') or ''}"
    print(f"{args.repo}#{issue['number']} @ {sha[:7]}: {issue['title'][:100]}", flush=True)

    embedder = EndpointEmbedder(ENDPOINT_MODEL, cache_dir=Path(args.cache))
    rep = live.report(snapshot, sha, args.repo, text, embedder, databricks_llm(), k=args.k)
    embedder.save()
    body = format_comment(rep)
    print(f"source {rep['source']}, tokens {rep['prompt_tokens']}+{rep['completion_tokens']}, "
          f"warnings {rep['warnings']}", flush=True)
    if args.dry_run:
        print("\n" + body)
    else:
        print("commented:", post(args.repo, issue["number"], body, os.environ["GITHUB_TOKEN"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
