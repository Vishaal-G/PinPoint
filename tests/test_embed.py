import numpy as np

from pinpoint.embed import QUERY_PREFIX, EndpointEmbedder, make_embedder


class FakeClient:
    """Stands in for WorkspaceClient: returns a 2-d vector per input, shuffled, and records batches."""
    def __init__(self):
        self.batches = []
        self.api_client = self

    def do(self, method, path, body):
        self.batches.append(body["input"])
        data = [{"index": i, "embedding": [len(t), 1.0]} for i, t in enumerate(body["input"])]
        return {"data": data[::-1]}  # out of order on purpose


def test_endpoint_embedder_batches_caches_normalizes_and_saves(tmp_path):
    client = FakeClient()
    e = EndpointEmbedder("databricks-x", cache_dir=tmp_path, client=client)
    e.batch_size = 2
    v = e.encode(["a", "bbb", "a", "cc"])
    assert client.batches == [["a", "bbb"], ["cc"]]          # duplicates sent once
    assert np.allclose(np.linalg.norm(v, axis=1), 1)
    assert np.allclose(v[0], v[2]) and v[1][0] > v[0][0]       # order restored by index
    e.encode(["bbb"])
    assert len(client.batches) == 2 and e.hits == 2 and e.misses == 3
    e.save()
    again = EndpointEmbedder("databricks-x", cache_dir=tmp_path, client=FakeClient())
    assert np.allclose(again.encode(["cc"]), v[3]) and again.misses == 0


def test_endpoint_query_gets_the_bge_prefix(tmp_path):
    client = FakeClient()
    EndpointEmbedder("databricks-x", cache_dir=tmp_path, client=client).encode_query("bug")
    assert client.batches == [[QUERY_PREFIX + "bug"]]


def test_make_embedder_picks_endpoint_by_name(tmp_path):
    assert isinstance(make_embedder("databricks-x", cache_dir=tmp_path, client=FakeClient()), EndpointEmbedder)


def test_endpoint_embedder_accepts_a_string_cache_dir(tmp_path):
    e = EndpointEmbedder("databricks-x", cache_dir=str(tmp_path), client=FakeClient())  # tempfile.mkdtemp() is a str
    e.encode(["a"])
    e.save()
    assert (tmp_path / "embeddings_databricks-x.parquet").exists()


class FakeTextEmbedding:
    """Stands in for fastembed.TextEmbedding: un-normalised vectors from the text length, records calls."""
    embedding_size = 3

    def __init__(self, model_name, cache_dir=None):
        self.calls = []

    def embed(self, texts, batch_size=32):
        self.calls.append(list(texts))
        return (np.array([len(t), 1.0, 2.0]) for t in texts)


def local(tmp_path, monkeypatch):
    import fastembed

    from pinpoint.embed import Embedder

    monkeypatch.setattr(fastembed, "TextEmbedding", FakeTextEmbedding)
    return Embedder(cache_dir=tmp_path)


def test_local_embedder_caches_and_normalises(tmp_path, monkeypatch):
    e = local(tmp_path, monkeypatch)
    v = e.encode(["a", "bb", "a"])
    assert v.shape == (3, 3) and np.allclose(np.linalg.norm(v, axis=1), 1)
    assert e.model.calls == [["a", "bb"]] and (e.hits, e.misses) == (1, 2)  # duplicate run once
    e.save()
    again = local(tmp_path, monkeypatch)
    assert np.allclose(again.encode(["bb"]), v[1:2]) and again.model.calls == []  # from the parquet
    assert again.encode([]).shape == (0, 3)


def test_local_query_gets_prefix_and_is_not_cached(tmp_path, monkeypatch):
    e = local(tmp_path, monkeypatch)
    e.encode_query("bug")
    assert e.model.calls == [[QUERY_PREFIX + "bug"]] and e.cache == {}


def test_default_embedder_env_switch(tmp_path, monkeypatch):
    import pinpoint.embed as embed

    monkeypatch.setattr(embed, "EndpointEmbedder", lambda *a, **kw: "endpoint")
    monkeypatch.setattr(embed, "Embedder", lambda *a, **kw: "local")
    monkeypatch.delenv("PINPOINT_EMBEDDER", raising=False)
    assert embed.default_embedder(tmp_path) == "endpoint"
    monkeypatch.setenv("PINPOINT_EMBEDDER", "local")
    assert embed.default_embedder(tmp_path) == "local"
