import pytest
import json
import sys
import types

from git import Repo

from pinpoint import bot
from test_agent import reply, scripted
from test_api import repo  # noqa: F401  (fixture: two-author git repo)
from test_live import FakeEmbedder

SHA = "abcdef1234567890"
REPORT = {
    "repo": "o/r", "commit": SHA, "source": "agent", "warnings": [], "llm": "Llama 3.3 70B",
    "suspects": [{"function_id": "pkg/mod.py::beta", "path": "pkg/mod.py", "start_line": 5, "end_line": 6,
                  "owner": {"name": "Bob", "email": "bob@x.org", "lines": 1, "share": 0.5}},
                 {"function_id": "pkg/mod.py::alpha", "path": "pkg/mod.py", "start_line": 1, "end_line": 2,
                  "owner": None}],
    "explanation": {"text": "beta raises at pkg/mod.py:6, not pkg/mod.py:40. cc @alice",
                    "citations": [{"path": "pkg/mod.py", "line": 6, "valid": True},
                                  {"path": "pkg/mod.py", "line": 40, "valid": False}]},
}


def test_comment_links_suspects_and_only_valid_citations():
    body = bot.format_comment(REPORT)
    assert body.startswith(bot.MARKER)
    assert f"[`beta`](https://github.com/o/r/blob/{SHA}/pkg/mod.py#L5-L6)" in body
    assert "Bob (50% of lines)" in body and "| unknown |" in body
    assert f"[`pkg/mod.py:6`](https://github.com/o/r/blob/{SHA}/pkg/mod.py#L6)" in body
    assert "pkg/mod.py:40" in body and "#L40" not in body          # invalid citation stays plain text
    assert "@alice" not in body and "@​alice" in body           # no pinging from model output
    assert "agent (dense retrieval" in body and "Warnings" not in body


def test_comment_shows_fallback_and_warnings():
    body = bot.format_comment({**REPORT, "source": "dense", "explanation": None, "warnings": ["no explanation: down"]})
    assert "dense retrieval only" in body and "**Why:**" not in body
    assert "<details><summary>Warnings</summary>" in body and "- no explanation: down" in body


class FakeComment:
    def __init__(self, body, bot_user=True):
        self.body, self.user = body, types.SimpleNamespace(type="Bot" if bot_user else "User")
        self.html_url = "https://github.com/o/r/issues/1#c1"

    def edit(self, body):
        self.body = body


def fake_github(monkeypatch, comments):
    created = []

    class Issue:
        def get_comments(self):
            return comments

        def create_comment(self, body):
            created.append(body)
            return types.SimpleNamespace(html_url="https://github.com/o/r/issues/1#new")

    class Github:
        def __init__(self, auth):
            assert auth.token == "tok"

        def get_repo(self, name):
            assert name == "o/r"
            return types.SimpleNamespace(get_issue=lambda n: Issue())

    monkeypatch.setitem(sys.modules, "github", types.SimpleNamespace(
        Github=Github, Auth=types.SimpleNamespace(Token=lambda t: types.SimpleNamespace(token=t))))
    return created


def test_post_edits_our_earlier_comment(monkeypatch):
    ours, theirs = FakeComment(f"{bot.MARKER} old"), FakeComment(f"quoting {bot.MARKER}", bot_user=False)
    created = fake_github(monkeypatch, [theirs, ours])
    assert bot.post("o/r", 1, "new body", "tok") == ours.html_url
    assert (ours.body, theirs.body, created) == ("new body", f"quoting {bot.MARKER}", [])


def test_post_creates_when_none(monkeypatch):
    created = fake_github(monkeypatch, [FakeComment("unrelated", bot_user=False)])
    assert bot.post("o/r", 1, "body", "tok").endswith("#new") and created == ["body"]


def event_file(tmp_path, user_type="User"):
    path = tmp_path.parent / f"{tmp_path.name}-event.json"   # outside the checkout
    path.write_text(json.dumps({"issue": {"number": 7, "title": "it crashes", "body": "boom",
                                          "user": {"type": user_type}}}))
    return path


def test_main_dry_run_end_to_end(repo, monkeypatch, capsys):  # noqa: F811
    import pinpoint.agent
    import pinpoint.embed

    call, bodies = scripted(reply(("submit_ranking", {"functions": ["pkg/mod.py::beta"]})),
                            reply(content="beta raises at pkg/mod.py:6."))
    monkeypatch.setattr(pinpoint.agent, "databricks_llm", lambda: call)
    monkeypatch.setattr(pinpoint.embed, "EndpointEmbedder", lambda *a, **kw: FakeEmbedder())
    assert bot.main(["--event", str(event_file(repo)), "--repo", "o/r", "--workspace", str(repo), "--dry-run"]) == 0
    out = capsys.readouterr().out
    sha = Repo(repo).head.commit.hexsha
    assert f"o/r#7 @ {sha[:7]}: it crashes" in out
    assert f"[`beta`](https://github.com/o/r/blob/{sha}/pkg/mod.py#L5-L6)" in out and "Bob (50% of lines)" in out
    assert "it crashes\nboom" in bodies[0]["messages"][1]["content"]       # title + body reach the agent


def test_main_skips_bot_issues(repo, capsys):  # noqa: F811
    assert bot.main(["--event", str(event_file(repo, "Bot")), "--repo", "o/r", "--workspace", str(repo)]) == 0
    assert "skipping" in capsys.readouterr().out


def test_backticked_citation_is_linked_outside_the_code_span():
    ex = {"text": "see `pkg/mod.py:6` and `pkg/mod.py:40`", "citations": [
        {"path": "pkg/mod.py", "line": 6, "valid": True}, {"path": "pkg/mod.py", "line": 40, "valid": False}]}
    out = bot.linkify(ex, "o/r", SHA)
    assert out == f"see [`pkg/mod.py:6`](https://github.com/o/r/blob/{SHA}/pkg/mod.py#L6) and `pkg/mod.py:40`"


def test_backticked_range_citation_links_the_range():
    # seen live on issue #1: the model wrote `src/pinpoint/bot.py:48-50`
    ex = {"text": "goes in `pkg/mod.py:5-6` and pkg/mod.py:6-7.", "citations": [
        {"path": "pkg/mod.py", "line": 5, "valid": True}, {"path": "pkg/mod.py", "line": 6, "valid": True}]}
    url = f"https://github.com/o/r/blob/{SHA}/pkg/mod.py"
    assert bot.linkify(ex, "o/r", SHA) == f"goes in [`pkg/mod.py:5-6`]({url}#L5-L6) and [`pkg/mod.py:6-7`]({url}#L6-L7)."


@pytest.mark.parametrize("raw,bad", [
    ("see #1 and #4051", ["#1", "#4051"]),
    ("fixed upstream in pytest-dev/pytest#12", ["pytest-dev/pytest#12"]),
    ("dup of GH-7", ["GH-7"]),
    ("https://github.com/o/r/issues/5 and https://github.com/o/r/pull/6", ["github.com/o/r/issues/5", "github.com/o/r/pull/6"]),
    ("cc @alice", ["@alice"]),
])
def test_defang_breaks_pings_and_cross_references(raw, bad):
    out = bot.defang(raw)
    assert out.replace(bot.ZWSP, "") == raw            # looks identical to a reader
    assert not any(b in out for b in bad)


@pytest.mark.parametrize("raw", ["color #fff", "line 12# comment", "C# and F#", "a#b", "https://github.com/o/r/blob/x/a.py#L3"])
def test_defang_leaves_non_references_alone(raw):
    assert bot.defang(raw) == raw


def test_issue_1_comment_has_no_live_reference():
    rep = {**REPORT, "explanation": {"text": "Same as #1, see https://github.com/o/r/issues/1.", "citations": []}}
    body = bot.format_comment(rep)
    assert "#1" not in body.replace(f"#{bot.ZWSP}1", "") and "github.com/o/r/issues" not in body


def test_footer_names_the_configured_llm_or_says_there_is_none():
    assert "agent (dense retrieval + Llama 3.3 70B)" in bot.format_comment(REPORT)
    assert "agent (dense retrieval + qwen3:32b)" in bot.format_comment({**REPORT, "llm": "qwen3:32b"})
    fallback = bot.format_comment({**REPORT, "source": "dense", "explanation": None})
    assert "dense retrieval only at" in fallback                      # LLM configured but down
    none = bot.format_comment({**REPORT, "source": "dense", "explanation": None, "llm": None})
    assert "dense retrieval only (no LLM configured)" in none


def test_main_with_no_llm_needs_no_databricks(repo, monkeypatch, capsys):  # noqa: F811
    import pinpoint.agent
    import pinpoint.embed

    def no_databricks(*a, **kw):
        raise AssertionError("Databricks must not be touched")
    monkeypatch.setattr(pinpoint.agent, "databricks_llm", no_databricks)
    monkeypatch.setattr(pinpoint.embed, "EndpointEmbedder", no_databricks)
    monkeypatch.setattr(pinpoint.embed, "Embedder", lambda *a, **kw: FakeEmbedder())
    monkeypatch.setenv("PINPOINT_LLM", "none")
    monkeypatch.setenv("PINPOINT_EMBEDDER", "local")
    assert bot.main(["--event", str(event_file(repo)), "--repo", "o/r", "--workspace", str(repo), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "[`beta`]" in out and "(no LLM configured)" in out
    assert "Warnings" not in out and "**Why:**" not in out


def test_index_only_fills_the_cache_without_an_issue(repo, monkeypatch, capsys):  # noqa: F811
    import pinpoint.embed

    fake = FakeEmbedder()
    monkeypatch.setattr(pinpoint.embed, "EndpointEmbedder", lambda *a, **kw: fake)
    assert bot.main(["--index-only", "--repo", "o/r", "--workspace", str(repo)]) == 0   # no --event
    assert fake.misses > 0 and fake.saves >= 1
    assert "embedded this run" in capsys.readouterr().out
