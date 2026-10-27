import pandas as pd

from pinpoint.graph import expand, snapshot_graph

# run calls is_rewrite_disabled (unique name) and __init__ (two functions have that short name).
FNS = pd.DataFrame({
    "function_id": ["r.py::A.run", "r.py::A.is_rewrite_disabled", "r.py::A.__init__", "s.py::B.__init__", "s.py::other"],
    "qualname": ["A.run", "A.is_rewrite_disabled", "A.__init__", "B.__init__", "other"],
    "hash": ["h_run", "h_dis", "h_ia", "h_ib", "h_other"],
})
EDGES = pd.DataFrame({"hash": ["h_run", "h_run", "h_dis"], "callee": ["is_rewrite_disabled", "__init__", "len"]})


def test_edges_resolve_by_short_name_with_ambiguity_weights():
    g = snapshot_graph(FNS, EDGES)
    assert g["callees"]["r.py::A.run"] == {"r.py::A.is_rewrite_disabled": 1.0, "r.py::A.__init__": 0.5, "s.py::B.__init__": 0.5}
    assert g["callers"]["r.py::A.is_rewrite_disabled"] == {"r.py::A.run": 1.0}
    assert "r.py::A.is_rewrite_disabled" not in g["callees"]   # `len` matches no repo function


def test_unweighted_graph_treats_every_target_alike():
    assert snapshot_graph(FNS, EDGES, weighted=False)["callees"]["r.py::A.run"]["s.py::B.__init__"] == 1.0


RANKING = ["r.py::A.is_rewrite_disabled", "s.py::other", "s.py::B.__init__", "r.py::A.run", "r.py::A.__init__"]


def test_uncapped_caller_can_overtake_its_seed():
    # The failing frame ranks first; its only caller `run` (4th) gets a bonus as big as the seed's score.
    out = expand(RANKING, snapshot_graph(FNS, EDGES), k=1, direction="callers")
    assert out[:2] == ["r.py::A.run", "r.py::A.is_rewrite_disabled"]


def test_capped_caller_lands_right_after_its_seed():
    out = expand(RANKING, snapshot_graph(FNS, EDGES), k=1, direction="callers", cap=True)
    assert out[:3] == ["r.py::A.is_rewrite_disabled", "r.py::A.run", "s.py::other"]


def test_no_neighbours_means_no_change():
    ranking = ["s.py::other", "r.py::A.run"]
    assert expand(ranking, snapshot_graph(FNS, EDGES), k=1) == ranking


def test_direction_filters_neighbours():
    ranking = ["r.py::A.is_rewrite_disabled", "s.py::other", "r.py::A.run"]
    assert expand(ranking, snapshot_graph(FNS, EDGES), k=1, direction="callees") == ranking   # it calls nothing in-repo
