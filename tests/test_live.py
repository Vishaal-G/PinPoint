import numpy as np
import pytest

from pinpoint import live
from pinpoint.embed import text_hash
from test_agent import reply, scripted


class FakeEmbedder:
    """Same interface as EndpointEmbedder: a hash cache, a miss counter, unit vectors.
    Bodies mentioning 'crash' point one way, everything else another; the query points at 'crash'."""

    def __init__(self):
        self.cache, self.misses, self.saves = {}, 0, 0

    def encode(self, texts):
        for t in texts:
            if text_hash(t) not in self.cache:
                self.misses += 1
                self.cache[text_hash(t)] = np.array([1.0, 0.0]) if "crash" in t else np.array([0.0, 1.0])
        return np.stack([self.cache[text_hash(t)] for t in texts])

    def encode_query(self, text):
        return np.array([0.8, 0.6])

    def save(self):
        self.saves += 1


@pytest.fixture
def snap(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("def alpha():\n    return 1\n\n\ndef beta():\n    raise crash\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_mod.py").write_text("def test_beta():\n    crash\n")
    return tmp_path


def test_index_lists_source_functions_only(snap):
    fns = live.index(snap)
    assert fns.function_id.tolist() == ["pkg/mod.py::alpha", "pkg/mod.py::beta"]
    assert fns[["start_line", "end_line"]].values.tolist() == [[1, 2], [5, 6]]


def test_embed_only_sends_new_bodies_and_saves(snap):
    fns, emb = live.index(snap), FakeEmbedder()
    live.embed(fns, emb, save_every=1)
    assert (emb.misses, emb.saves) == (2, 2)
    live.embed(fns, emb)                 # second index of the same code: all cache hits, no save
    assert (emb.misses, emb.saves) == (2, 2)


def test_triage_seeds_with_dense_and_drops_invented_ids(snap):
    call, bodies = scripted(reply(("submit_ranking", {"functions": ["pkg/mod.py::beta", "pkg/mod.py::nope"]})))
    out = live.triage(snap, "o/r", "it crashes", FakeEmbedder(), call)
    assert out["seeds"] == ["pkg/mod.py::beta", "pkg/mod.py::alpha"]
    assert "1. pkg/mod.py::beta (lines 5-6)" in bodies[0]["messages"][1]["content"]   # seeds reach the agent
    assert "call_graph" in {t["function"]["name"] for t in bodies[0]["tools"]}          # H3 tools
    assert out["functions"] == [{"id": "pkg/mod.py::beta", "start_line": 5, "end_line": 6}]
    assert (out["source"], out["files"], out["invented"]) == ("agent", ["pkg/mod.py"], ["pkg/mod.py::nope"])


def test_triage_falls_back_to_dense_when_the_agent_names_nothing_real(snap):
    call, _ = scripted(reply(("submit_ranking", {"functions": ["made/up.py::thing"]})))
    out = live.triage(snap, "o/r", "it crashes", FakeEmbedder(), call)
    assert out["source"] == "dense"
    assert [f["id"] for f in out["functions"]] == ["pkg/mod.py::beta", "pkg/mod.py::alpha"]


def test_checkout_uses_a_local_directory_as_is(tmp_path):
    from git import Repo

    repo = Repo.init(tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n")
    repo.index.add(["a.py"])
    sha = repo.index.commit("one").hexsha
    assert live.checkout(str(tmp_path)) == (tmp_path, sha)
