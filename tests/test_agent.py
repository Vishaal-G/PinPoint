import json

import pytest

from pinpoint import agent


def reply(*calls, content=""):
    """A fake chat response whose message makes these (name, args) tool calls."""
    tool_calls = [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                  for i, (n, a) in enumerate(calls)]
    return {"choices": [{"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


def scripted(*replies):
    """call_llm that returns the replies in order and records each request body."""
    bodies, queue = [], list(replies)

    def call(body):
        bodies.append(json.loads(json.dumps(body)))  # snapshot: messages keeps growing
        return queue.pop(0)
    return call, bodies


@pytest.fixture
def snap(tmp_path):
    (tmp_path / "mod.py").write_text("def alpha():\n    return 1\n")
    return tmp_path


def test_loop_runs_tools_then_submits(snap):
    call, bodies = scripted(reply(("grep", {"pattern": "def alpha"})),
                            reply(("submit_ranking", {"functions": ["./mod.py::alpha", "mod.py:alpha", "mod.py::beta"]})))
    out = agent.run_agent(call, snap, "o/r", "issue text")
    assert out["stop"] == "submitted"
    assert out["ranking"] == ["mod.py::alpha", "mod.py::beta"]
    # The second request carries the first tool result, linked by tool_call_id.
    tool_msg = bodies[1]["messages"][-1]
    assert tool_msg == {"role": "tool", "tool_call_id": "c0", "content": "mod.py:1: def alpha():"}
    assert out["prompt_tokens"] == 200 and len(out["steps"]) == 2


def test_last_step_forces_submit(snap):
    call, bodies = scripted(reply(("grep", {"pattern": "x"})), reply(("grep", {"pattern": "y"})),
                            reply(("submit_ranking", {"functions": ["mod.py::alpha"]})))
    out = agent.run_agent(call, snap, "o/r", "issue", max_steps=3)
    assert [b["tool_choice"] for b in bodies[:2]] == ["auto", "auto"]
    assert bodies[2]["tool_choice"]["function"]["name"] == "submit_ranking"
    assert out["stop"] == "submitted"


def test_prose_reply_gets_a_nudge_and_bad_submit_an_error(snap):
    prose = {"choices": [{"message": {"role": "assistant", "content": "I think it's alpha."}}], "usage": {}}
    call, bodies = scripted(prose, reply(("submit_ranking", {"functions": "alpha"})),
                            reply(("submit_ranking", {"functions": ["mod.py::alpha"]})))
    out = agent.run_agent(call, snap, "o/r", "issue")
    assert bodies[1]["messages"][-1]["role"] == "user"
    assert bodies[2]["messages"][-1]["content"].startswith("error: bad submit_ranking")
    assert out["ranking"] == ["mod.py::alpha"]


def test_no_submit_within_limit_gives_empty_ranking(snap):
    call, _ = scripted(reply(("grep", {"pattern": "x"})), reply(("grep", {"pattern": "y"})))
    out = agent.run_agent(call, snap, "o/r", "issue", max_steps=2)
    assert out["stop"] == "max_steps" and out["ranking"] == []


def test_seeds_are_appended_to_the_issue(snap):
    call, bodies = scripted(reply(("submit_ranking", {"functions": ["mod.py::alpha"]})))
    agent.run_agent(call, snap, "o/r", "issue", seeds=[("mod.py::alpha", 1, 2)])
    user = bodies[0]["messages"][1]["content"]
    assert user.startswith("issue\n\nA retriever ranked") and user.endswith("1. mod.py::alpha (lines 1-2)")
    assert agent.PROMPT_VERSION_H != agent.PROMPT_VERSION


def test_trace_records_agent_steps_and_tools(snap, tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test")
    call, _ = scripted(reply(("grep", {"pattern": "def alpha"})),
                       reply(("submit_ranking", {"functions": ["mod.py::alpha"]})))
    agent.run_agent(call, snap, "o/r", "issue", seeds=[("mod.py::alpha", 1, 2)], trace=True, tags={"instance_id": "x"})
    mlflow.flush_trace_async_logging()  # traces are written in the background
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    spans = [(s.name, s.span_type) for s in trace.data.spans]
    assert spans == [("agent", "AGENT"), ("pinpoint_seeds", "RETRIEVER"), ("step 1", "CHAT_MODEL"),
                     ("grep", "TOOL"), ("step 2", "CHAT_MODEL"), ("submit_ranking", "TOOL")]
    assert trace.info.tags["instance_id"] == "x"
    step1 = trace.data.spans[2]
    assert step1.attributes["prompt_tokens"] == 100


def test_fixes_change_prompt_tools_and_version(snap):
    (snap / "cli.py").write_text("def addopts(p):\n    p.addoption('--flag')\n")
    call, bodies = scripted(reply(("grep", {"pattern": "addoption"})),
                            reply(("submit_ranking", {"functions": ["cli.py::addopts"]})))
    agent.run_agent(call, snap, "o/r", "fails with --flag", fixes=frozenset(agent.FIXES))
    system, user = bodies[0]["messages"][0]["content"], bodies[0]["messages"][1]["content"]
    assert "Advice from past mistakes" in system
    assert user.endswith("defined here:\n--flag: cli.py:2 [cli.py::addopts]")
    assert "labelled" in bodies[0]["tools"][0]["function"]["description"]
    assert bodies[1]["messages"][-1]["content"] == "cli.py:2 [cli.py::addopts]: p.addoption('--flag')"
    # No fixes keeps the original cache keys; each subset gets its own.
    assert agent.prompt_version() == agent.PROMPT_VERSION and agent.prompt_version(True) == agent.PROMPT_VERSION_H
    assert len({agent.prompt_version(True, frozenset(f)) for f in ([], ["labels"], ["labels", "prompt"])}) == 3


def test_parse_fixes():
    assert agent.parse_fixes(["x", "--fixes=labels,prompt"]) == {"labels", "prompt"}
    with pytest.raises(SystemExit):
        agent.parse_fixes(["--fixes=magic"])
