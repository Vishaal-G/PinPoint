from pathlib import Path

import pytest
import streamlit as st
from git import GitCommandError, Repo
from streamlit.testing.v1 import AppTest

from pinpoint import live, providers
from test_agent import reply, scripted
from test_api import repo  # noqa: F401  (fixture: two-author git repo)
from test_live import FakeEmbedder

APP = str(Path(__file__).parents[1] / "src" / "pinpoint" / "app.py")
GOOD = (reply(("submit_ranking", {"functions": ["pkg/mod.py::beta", "pkg/mod.py::alpha"]})),
        reply(content="The crash is raised in beta at pkg/mod.py:6, not pkg/mod.py:99."))


@pytest.fixture
def app(repo, monkeypatch, tmp_path_factory):
    """The app with a fake checkout (the two-author repo), fake embedder, and a settable fake LLM."""
    live_dir = tmp_path_factory.mktemp("live")
    for name in ["o__r", "o__missing", "..__etc"]:  # AppTest can't type a new option, so list them
        (live_dir / "repos" / name).mkdir(parents=True)
    monkeypatch.setattr(live, "LIVE_DIR", live_dir)
    sha = Repo(repo).head.commit.hexsha
    monkeypatch.setattr(live, "checkout", lambda name, ref=None: (repo, sha))
    emb = FakeEmbedder()
    monkeypatch.setattr(providers, "get_embedder", lambda: emb)
    st.cache_data.clear()  # the report cache is process-wide: don't leak between tests
    llm = {"calls": 0, "replies": []}

    def call(body):
        llm["calls"] += 1
        if not llm["replies"]:
            raise RuntimeError("BAD_REQUEST: Cannot create or query foundation model endpoints")
        return llm["replies"].pop(0)
    monkeypatch.setattr(providers, "get_llm", lambda: call)
    at = AppTest.from_file(APP, default_timeout=30).run()
    return at, llm, sha


def submit(at, repo="o/r", title="it crashes", body="", url=""):
    at.selectbox(key="repo").set_value(repo)
    at.text_input(key="url").set_value(url)
    at.text_input(key="title").set_value(title)
    at.text_area(key="body").set_value(body)
    return at.button[0].click().run()


def test_triage_shows_suspects_owner_and_linked_explanation(app):
    at, llm, sha = app
    llm["replies"] = list(GOOD)
    submit(at)
    assert not at.exception and not at.error, at.error
    assert [m.value for m in at.metric] == ["agent", sha[:7], "220"]
    df = at.dataframe[0].value
    assert df.function.tolist() == ["beta", "alpha"]
    assert df.owner.tolist() == ["Bob (50%)", "Alice (100%)"]
    assert df.code[0] == f"https://github.com/o/r/blob/{sha}/pkg/mod.py#L5-L6"
    why = at.markdown[-1].value if "pkg/mod.py:6" in at.markdown[-1].value else \
        next(m.value for m in at.markdown if "beta at" in m.value)
    assert f"[`pkg/mod.py:6`](https://github.com/o/r/blob/{sha}/pkg/mod.py#L6)" in why
    assert "pkg/mod.py:99" in why and "#L99" not in why                  # invalid citation stays plain
    assert "1/2 citations" in at.caption[-1].value
    assert at.warning[0].value == "explanation cites lines outside the code it was shown"


def test_same_issue_again_is_a_cache_hit(app):
    at, llm, _ = app
    llm["replies"] = list(GOOD)
    submit(at)
    submit(at)
    assert llm["calls"] == 2              # agent + explanation once; the second submit never reached the LLM
    assert at.metric[0].value == "agent"


def test_llm_down_shows_dense_with_warnings_and_is_not_cached(app):
    at, llm, _ = app
    submit(at)
    assert at.metric[0].value == "dense only"
    assert at.dataframe[0].value.function.tolist() == ["beta", "alpha"]
    assert len(at.warning) == 2 and "agent unavailable" in at.warning[0].value
    calls = llm["calls"]
    llm["replies"] = list(GOOD)          # LLM back: the next submit must recompute, not serve the fallback
    submit(at)
    assert llm["calls"] > calls and at.metric[0].value == "agent"


def test_embedding_failure_is_an_error(app, monkeypatch):
    at, _, _ = app

    class Down(FakeEmbedder):
        def encode(self, texts):
            raise RuntimeError("quota")
    monkeypatch.setattr(providers, "get_embedder", lambda: Down())
    submit(at)
    assert "embedding failed: quota" in at.error[0].value and not at.dataframe


def test_bad_checkout_is_an_error(app, monkeypatch):
    at, _, _ = app

    def fail(name, ref=None):
        raise GitCommandError("clone", 128)
    monkeypatch.setattr(live, "checkout", fail)
    submit(at, repo="o/missing")
    assert at.error[0].value == "cannot check out o/missing at the default branch"


def test_issue_url_fills_repo_and_text(app, monkeypatch):
    at, llm, _ = app
    llm["replies"] = list(GOOD)
    seen = []
    monkeypatch.setattr(live, "github_issue", lambda r, n: seen.append((r, n)) or "it crashes\nin beta")
    at.selectbox(key="repo").set_value(None)
    at.text_input(key="url").set_value("https://github.com/o/r/issues/7")
    at.button[0].click().run()
    assert seen == [("o/r", 7)] and not at.error
    assert at.dataframe[0].value.function.tolist() == ["beta", "alpha"]


def test_input_validation(app):
    at, llm, _ = app
    submit(at, repo="../etc")
    assert at.error[0].value == "Repository must look like owner/name."
    submit(at, title="")
    assert at.error[0].value == "Give an issue title/body or a GitHub issue URL."
    assert llm["calls"] == 0


def test_indexed_repos(tmp_path):
    from pinpoint.app import indexed_repos  # noqa: F401  (importing runs main(); fine in bare mode)
    (tmp_path / "pytest-dev__pytest").mkdir()
    (tmp_path / "a__b__c").mkdir()
    assert indexed_repos(tmp_path) == ["a/b__c", "pytest-dev/pytest"]
    assert indexed_repos(tmp_path / "none") == []


@pytest.mark.parametrize("name,ok", [("pytest-dev/pytest", True), ("o/.github", True), ("a/b.c-d_e", True),
                                     ("../etc", False), ("o/..", False), ("o/.", False), ("o", False),
                                     ("o/r/x", False), ("o_x/r", False)])
def test_repo_pattern(name, ok):
    import re
    assert bool(re.match(live.REPO_PATTERN, name)) is ok


def test_llm_client_connects_lazily(monkeypatch):
    import pinpoint.agent as agent
    providers.get_llm.cache_clear()
    built = []

    def broken():
        built.append(1)
        raise IOError("cannot get access token")
    monkeypatch.setattr(agent, "databricks_llm", broken)
    call = providers.get_llm()              # no connection yet: building the provider can't fail
    assert built == []
    with pytest.raises(IOError):
        call({"messages": []})             # fails at call time, inside the agent's try
    providers.get_llm.cache_clear()
