import pytest
from fastapi.testclient import TestClient
from git import Actor, GitCommandError, Repo

from pinpoint import api, live
from pinpoint.blame import owner
from pinpoint.explain import check_citations, explain
from test_agent import reply, scripted
from test_live import FakeEmbedder

ALICE, BOB = Actor("Alice", "alice@x.org"), Actor("Bob", "bob@x.org")


@pytest.fixture
def repo(tmp_path):
    """pkg/mod.py: alpha written by Alice; beta's body rewritten by Bob in a second commit."""
    r = Repo.init(tmp_path)
    (tmp_path / "pkg").mkdir()
    mod = tmp_path / "pkg" / "mod.py"
    mod.write_text("def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n")
    r.index.add(["pkg/mod.py"])
    r.index.commit("first", author=ALICE, committer=ALICE, author_date="2020-01-01T00:00:00")
    mod.write_text("def alpha():\n    return 1\n\n\ndef beta():\n    raise crash\n")
    r.index.add(["pkg/mod.py"])
    r.index.commit("second", author=BOB, committer=BOB, author_date="2021-01-01T00:00:00")
    return tmp_path


def test_owner_is_author_of_most_lines(repo):
    assert owner(repo, "pkg/mod.py", 1, 2) == {"name": "Alice", "email": "alice@x.org", "lines": 2, "share": 1.0}
    assert owner(repo, "pkg/mod.py", 5, 6) == {"name": "Bob", "email": "bob@x.org", "lines": 1, "share": 0.5}  # tie: newer wins
    assert owner(repo, "pkg/mod.py", 6, 6)["name"] == "Bob"
    assert owner(repo, "pkg/missing.py", 1, 2) is None


def test_owner_outside_git_is_none(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    assert owner(tmp_path, "a.py", 1, 1) is None


def test_citations_checked_against_shown_lines():
    shown = [("pkg/mod.py", 5, 6, "...")]
    assert check_citations("see pkg/mod.py:6, pkg/mod.py:40 and other.py:6; again pkg/mod.py:6", shown) == [
        {"path": "pkg/mod.py", "line": 6, "valid": True},
        {"path": "pkg/mod.py", "line": 40, "valid": False},
        {"path": "other.py", "line": 6, "valid": False}]


def test_explain_shows_numbered_code_and_checks_citations(repo):
    call, bodies = scripted(reply(content="beta raises at pkg/mod.py:6."))
    ex = explain(call, "it crashes", repo, [{"id": "pkg/mod.py::beta", "start_line": 5, "end_line": 6}])
    assert "    6| " in bodies[0]["messages"][1]["content"]
    assert (ex["text"], ex["citations"]) == ("beta raises at pkg/mod.py:6.", [{"path": "pkg/mod.py", "line": 6, "valid": True}])
    assert (ex["prompt_tokens"], ex["completion_tokens"]) == (100, 10)


@pytest.fixture
def client(repo, monkeypatch):
    sha = Repo(repo).head.commit.hexsha
    monkeypatch.setattr(live, "checkout", lambda name, ref=None: (repo, sha))
    api.app.dependency_overrides[api.get_embedder] = FakeEmbedder
    use_llm(scripted()[0])  # default: never build the real Databricks client (needs a live login)
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def use_llm(call):
    api.app.dependency_overrides[api.get_llm] = lambda: call


def test_triage_end_to_end(client):
    call, _ = scripted(reply(("submit_ranking", {"functions": ["pkg/mod.py::beta", "pkg/mod.py::alpha"]})),
                       reply(content="The crash is raised in beta at pkg/mod.py:6."))
    use_llm(call)
    r = client.post("/triage", json={"repo": "o/r", "title": "it crashes", "k": 1})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["source"] == "agent" and out["warnings"] == []
    assert out["suspects"] == [{"function_id": "pkg/mod.py::beta", "path": "pkg/mod.py", "start_line": 5,
                                "end_line": 6, "owner": {"name": "Bob", "email": "bob@x.org", "lines": 1, "share": 0.5}}]
    assert out["explanation"]["citations"] == [{"path": "pkg/mod.py", "line": 6, "valid": True}]
    assert (out["prompt_tokens"], out["completion_tokens"]) == (200, 20)


def test_llm_down_falls_back_to_dense(client):
    def down(body):
        raise RuntimeError("BAD_REQUEST: Cannot create or query foundation model endpoints")
    use_llm(down)
    out = client.post("/triage", json={"repo": "o/r", "title": "it crashes"}).json()
    assert out["source"] == "dense"
    assert [s["function_id"] for s in out["suspects"]] == ["pkg/mod.py::beta", "pkg/mod.py::alpha"]
    assert out["explanation"] is None
    assert len(out["warnings"]) == 2   # agent unavailable + no explanation


def test_invalid_citation_warns(client):
    call, _ = scripted(reply(("submit_ranking", {"functions": ["pkg/mod.py::beta"]})),
                       reply(content="Look at pkg/mod.py:99."))
    use_llm(call)
    out = client.post("/triage", json={"repo": "o/r", "title": "it crashes"}).json()
    assert out["explanation"]["citations"][0]["valid"] is False
    assert out["warnings"] == ["explanation cites lines outside the code it was shown"]


def test_bad_request_is_422(client):
    assert client.post("/triage", json={"repo": "../etc", "title": ""}).status_code == 422


def test_unknown_repo_is_404(client, monkeypatch):
    def fail(name, ref=None):
        raise GitCommandError("clone", 128)
    monkeypatch.setattr(live, "checkout", fail)
    use_llm(scripted()[0])
    assert client.post("/triage", json={"repo": "no/such", "title": "x"}).status_code == 404
