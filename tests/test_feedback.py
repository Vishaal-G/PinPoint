import csv
import types
from datetime import datetime, timezone

from pinpoint import bot, feedback
from test_bot import REPORT

T = datetime(2026, 11, 2, tzinfo=timezone.utc)


def user(login, kind="User"):
    return types.SimpleNamespace(login=login, type=kind)


def comment(body, reactions, issue=1, author="Bot"):
    return types.SimpleNamespace(
        body=body, user=user("pinpoint[bot]", author), html_url=f"https://github.com/o/r/issues/{issue}#c",
        issue_url=f"https://api.github.com/repos/o/r/issues/{issue}", created_at=T, updated_at=T,
        get_reactions=lambda: [types.SimpleNamespace(content=c, user=u) for c, u in reactions])


class FakeRepo:
    def __init__(self, comments):
        self.comments = comments

    def get_issues_comments(self):
        return self.comments

    def get_issue(self, n):
        return types.SimpleNamespace(state="open", user=user("alice"))


def test_meta_round_trips_and_stays_inside_the_html_comment():
    rep = {**REPORT, "suspects": [{**REPORT["suspects"][0], "function_id": "a-->b.py::f"}, *REPORT["suspects"][1:]]}
    body = bot.format_comment(rep)
    line = body.splitlines()[1]
    assert line.startswith("<!-- pinpoint-meta") and line.count("-->") == 1   # the id's `-->` can't close it early
    assert bot.read_meta(body) == {"v": 1, "commit": REPORT["commit"], "source": "agent",
                                   "suspects": ["a-->b.py::f", "pkg/mod.py::alpha"]}
    assert bot.read_meta("no meta here") is None


def test_rows_count_human_votes_on_our_comments_only():
    ours = comment(bot.format_comment(REPORT), [("+1", user("alice")), ("+1", user("bob")), ("-1", user("carol")),
                                                ("+1", user("dependabot", "Bot")), ("heart", user("dan"))])
    old = comment(f"{bot.MARKER}\nbefore the meta line", [], issue=2)          # no meta: still counted
    spoof = comment(bot.format_comment(REPORT), [("+1", user("eve"))], issue=3, author="User")
    other = comment("unrelated bot comment", [("+1", user("alice"))], issue=4)
    rs = feedback.rows(FakeRepo([ours, old, spoof, other]), now=T)

    assert [r["issue"] for r in rs] == [1, 2]
    r = rs[0]
    assert (r["up"], r["down"], r["voters"], r["author_voted"]) == (2, 1, "alice bob carol", True)
    assert (r["source"], r["top_suspect"], r["n_suspects"]) == ("agent", "pkg/mod.py::beta", 2)
    assert (rs[1]["up"], rs[1]["top_suspect"], rs[1]["author_voted"]) == (0, "", False)

    s = feedback.summarize(rs)
    assert s == {"comments": 2, "rated": 1, "coverage": 0.5, "up": 2, "down": 1, "approval": 2 / 3}
    md = feedback.markdown(s, rs)
    assert "coverage 50%" in md and "approval 67%" in md and "Fewer than 20" in md


def test_summary_with_no_votes():
    assert feedback.summarize([])["approval"] is None
    assert "approval n/a" in feedback.markdown(feedback.summarize([]), [])


def test_main_writes_csv(monkeypatch, tmp_path):
    import sys
    repo = FakeRepo([comment(bot.format_comment(REPORT), [("-1", user("alice"))])])
    gh = types.SimpleNamespace(Github=lambda auth=None: types.SimpleNamespace(get_repo=lambda n: repo),
                               Auth=types.SimpleNamespace(Token=lambda t: t))
    monkeypatch.setitem(sys.modules, "github", gh)
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    out = tmp_path / "f.csv"
    assert feedback.main(["--repo", "o/r", "--out", str(out)]) == 0
    row = next(csv.DictReader(out.open()))
    assert (row["down"], row["top_suspect"]) == ("1", "pkg/mod.py::beta")
    assert "approval 0%" in summary.read_text()
