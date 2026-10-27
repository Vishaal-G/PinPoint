import json

import pandas as pd
import pytest

from pinpoint import llm_query
from pinpoint.llm_query import parse_arguments, rewrite

ANSWER = {"summary": "showfixtures ignores fixture scope", "functions": ["_showfixtures_main"],
          "files": ["src/_pytest/python.py"], "terms": ["FixtureDef.scope"]}


def test_parse_arguments_accepts_schema_and_fills_missing_lists():
    assert parse_arguments(json.dumps(ANSWER)) == ANSWER
    assert parse_arguments('{"summary": "x"}') == {"summary": "x", "functions": [], "files": [], "terms": []}


@pytest.mark.parametrize("raw", ['{"functions": []}', '{"summary": "x", "functions": "run"}', '[1, 2]'])
def test_parse_arguments_rejects_bad_shapes(raw):
    with pytest.raises(ValueError):
        parse_arguments(raw)


def test_rewrite_modes():
    text = "--fixtures should show scope\nbody"
    assert rewrite(text, ANSWER, "names") == "_showfixtures_main src/_pytest/python.py"
    assert rewrite(text, ANSWER, "raw+llm").startswith(text + "\n_showfixtures_main")
    assert rewrite(text, ANSWER, "title+llm") == "--fixtures should show scope\nshowfixtures ignores fixture scope\n_showfixtures_main src/_pytest/python.py"
    assert rewrite(text, {**ANSWER, "functions": [], "files": []}, "names") == text


class FakeClient:
    """Stands in for WorkspaceClient: first answer has no tool call, second is valid."""
    def __init__(self, replies):
        self.replies, self.api_client = list(replies), self

    def do(self, method, path, body):
        return self.replies.pop(0)


def reply(args):
    calls = [{"function": {"name": "submit_query", "arguments": json.dumps(args)}}] if args else None
    return {"choices": [{"message": {"tool_calls": calls}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def test_ask_retries_once_on_missing_tool_call():
    args, usage = llm_query.ask(FakeClient([reply(None), reply(ANSWER)]), "pytest-dev/pytest", "issue")
    assert args == ANSWER and usage["prompt_tokens"] == 10


def test_ask_gives_up_after_two_bad_answers():
    with pytest.raises(ValueError):
        llm_query.ask(FakeClient([reply(None), reply(None)]), "pytest-dev/pytest", "issue")


def test_cache_is_reused(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_query, "CACHE", tmp_path / "q.parquet")
    issues = pd.DataFrame({"instance_id": ["a"], "repo": ["r/r"], "problem_statement": ["t"]})
    pd.DataFrame([{"instance_id": "a", "model": llm_query.MODEL, "prompt_version": llm_query.PROMPT_VERSION,
                   **ANSWER, "prompt_tokens": 1, "completion_tokens": 1}]).to_parquet(llm_query.CACHE)
    out = llm_query.llm_queries(issues)   # would need a real endpoint if the cache were missed
    assert out.loc[0, "summary"] == ANSWER["summary"]
