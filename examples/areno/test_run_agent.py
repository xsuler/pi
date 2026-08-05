from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest


class _Turn:
    def __init__(self, *, item, messages, response):
        self.item = item
        self.messages = messages
        self.response = response
        metadata = response["areno"]
        self.input_tokens = metadata.get("input_tokens", [])
        self.response_tokens = metadata["response_tokens"]
        self.response_logprobs = metadata["response_logprobs"]


class _Trajectory:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _load_module():
    tools = types.ModuleType("areno.agent.tools")
    tools.CodingWorkspace = object
    agentic = types.ModuleType("areno.api.agentic")
    agentic.AgentTrajectory = _Trajectory
    agentic.AgentTrajectoryTurn = _Turn
    modules = {
        "areno": types.ModuleType("areno"),
        "areno.agent": types.ModuleType("areno.agent"),
        "areno.agent.tools": tools,
        "areno.api": types.ModuleType("areno.api"),
        "areno.api.agentic": agentic,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        path = Path(__file__).with_name("run_agent.py")
        spec = importlib.util.spec_from_file_location("pi_areno_run_agent", path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _load_rollout_module():
    path = Path(__file__).with_name("test_rollout.py")
    spec = importlib.util.spec_from_file_location("pi_areno_test_rollout", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _event(message):
    return json.dumps({"type": "message_end", "message": message})


def test_events_skip_non_trainable_assistant_messages():
    module = _load_module()
    lines = [
        _event({"role": "assistant", "content": [{"type": "text", "text": "internal"}]}),
        _event(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "call_1",
                        "name": "write",
                        "arguments": {"path": "index.html", "content": "ok"},
                    }
                ],
                "providerMetadata": {
                    "areno": {"input_tokens": [1], "response_tokens": [2], "response_logprobs": [-0.2]}
                },
            }
        ),
    ]

    turns = module._events_to_turns(object(), lines)

    assert len(turns) == 1
    assert turns[0].response_tokens == [2]
    assert turns[0].input_tokens == [1]


def test_events_preserve_tool_calls_and_finish_reason():
    module = _load_module()
    lines = [
        _event(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "call_1",
                        "name": "write",
                        "arguments": {"path": "index.html", "content": "<main>ok</main>"},
                    }
                ],
                "providerMetadata": {
                    "areno": {"input_tokens": [1], "response_tokens": [2], "response_logprobs": [-0.2]}
                },
            }
        )
    ]

    turns = module._events_to_turns(object(), lines)

    choice = turns[0].response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "write"


def test_areno_metadata_accepts_wire_compatible_top_level_field():
    module = _load_module()
    metadata = {"input_tokens": [1], "response_tokens": [2], "response_logprobs": [-0.2]}

    assert module._areno_metadata({"areno": metadata}) == metadata


def test_events_report_pi_model_error_message():
    module = _load_module()
    lines = [
        _event(
            {
                "role": "assistant",
                "content": [],
                "stopReason": "error",
                "errorMessage": "HTTP 400: invalid request body",
            }
        )
    ]

    with pytest.raises(RuntimeError, match="HTTP 400: invalid request body"):
        module._events_to_turns(object(), lines)


def test_run_agent_isolates_workspace_per_expanded_record(tmp_path):
    module = _load_module()
    created = []
    seen_roots = []

    class Workspace:
        def __init__(self, task, root):
            self.task = task
            self.root = root
            self.close_count = 0

        @classmethod
        def from_task(cls, task):
            workspace = cls(task, tmp_path / f"sample-{len(created)}")
            created.append(workspace)
            return workspace

        def close(self):
            self.close_count += 1

    async def fake_run_item(ctx, item, workspace):
        del ctx, item
        seen_roots.append(workspace.root)
        await asyncio.sleep(0)
        return []

    shared_record = {"id": "task-1", "files": {"README.md": "seed"}}
    items = [types.SimpleNamespace(record=shared_record), types.SimpleNamespace(record=shared_record)]
    batch = types.SimpleNamespace(iter_samples=lambda: iter(items))
    ctx = types.SimpleNamespace(max_running_prompts=2)
    module.CodingWorkspace = Workspace
    module._run_item = fake_run_item

    trajectory = asyncio.run(module.run_agent(ctx, batch))

    assert trajectory.turns == []
    assert len(set(seen_roots)) == 2
    assert all(workspace.close_count == 1 for workspace in created)
    assert created[0].task is not created[1].task


def test_rollout_summary_reports_tool_calls_and_generated_files(tmp_path, capsys):
    module = _load_rollout_module()
    (tmp_path / "index.html").write_text("<main>ok</main>", encoding="utf-8")
    events = [
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "stopReason": "toolUse",
                "content": [{"type": "toolCall", "name": "write", "arguments": {"path": "index.html"}}],
                "providerMetadata": {
                    "areno": {"input_tokens": [1, 2], "response_tokens": [3], "response_logprobs": [-0.1]}
                },
            },
        }
    ]
    result = subprocess.CompletedProcess(args=["pi"], returncode=0, stdout="", stderr="")

    trainable, tool_calls = module._print_summary(events, tmp_path, result)

    output = capsys.readouterr().out
    assert (trainable, tool_calls) == (1, 1)
    assert "assistant stop='toolUse' input_tokens=2 response_tokens=1 tool_calls=1" in output
    assert "file index.html: present" in output
