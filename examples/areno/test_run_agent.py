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


def test_training_explicitly_enables_every_builtin_pi_tool():
    module = _load_module()

    assert module.ALL_PI_TOOLS.split(",") == ["read", "bash", "edit", "write", "grep", "find", "ls"]


def test_adapter_and_single_rollout_use_identical_model_config(tmp_path):
    adapter = _load_module()
    rollout = _load_rollout_module()
    adapter_dir = tmp_path / "adapter"
    rollout_dir = tmp_path / "rollout"

    adapter._write_models(adapter_dir, "http://127.0.0.1:3000/v1", "test-key")
    rollout._write_models(rollout_dir, "http://127.0.0.1:3000/v1", "test-key")

    adapter_config = json.loads((adapter_dir / "models.json").read_text(encoding="utf-8"))
    rollout_config = json.loads((rollout_dir / "models.json").read_text(encoding="utf-8"))
    assert adapter_config == rollout_config


def test_generated_files_must_all_exist_and_be_nonempty(tmp_path):
    module = _load_module()
    (tmp_path / "index.html").write_text("<main>ok</main>", encoding="utf-8")
    (tmp_path / "styles.css").write_text("body{}", encoding="utf-8")
    (tmp_path / "app.js").write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="app.js"):
        module._validate_generated_files(tmp_path)

    (tmp_path / "app.js").write_text("void 0;", encoding="utf-8")
    module._validate_generated_files(tmp_path)


def test_stage_one_prompt_requests_compact_separate_files():
    module = _load_module()
    item = types.SimpleNamespace(prompt="Build a simple stock list.", record={"stage": 1})

    prompt = module._prompt(item)

    assert "three compact files" in prompt
    assert "Write each file separately" in prompt
    assert "Never run rm, rm -rf, or rmdir" in prompt
    assert "leave all three required files present" in prompt
    assert "polished responsive implementation" not in prompt


def test_standard_prompt_forbids_removing_generated_files():
    module = _load_module()
    item = types.SimpleNamespace(prompt="Build a dashboard.", record={})

    prompt = module._prompt(item)

    assert "Never run rm, rm -rf, or rmdir" in prompt
    assert "leave all three required files present" in prompt


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

        def close(self):
            self.close_count += 1

    async def fake_run_item(ctx, item, workspace):
        del ctx, item
        seen_roots.append(workspace.root)
        await asyncio.sleep(0)
        return []

    shared_record = {"id": "task-1", "files": {"README.md": "seed"}}
    items = [
        types.SimpleNamespace(record=shared_record, sample_index=0),
        types.SimpleNamespace(record=shared_record, sample_index=1),
    ]
    batch = types.SimpleNamespace(iter_samples=lambda: iter(items))
    ctx = types.SimpleNamespace(max_running_prompts=2)
    module.CodingWorkspace = Workspace
    def empty_workspace(item):
        workspace = Workspace(dict(item.record), tmp_path / f"sample-{item.sample_index}")
        created.append(workspace)
        return workspace

    module._empty_workspace = empty_workspace
    module._run_item = fake_run_item

    trajectory = asyncio.run(module.run_agent(ctx, batch))

    assert trajectory.turns == []
    assert trajectory.invalid_items == []
    assert len(set(seen_roots)) == 2
    assert all(workspace.close_count == 1 for workspace in created)
    assert created[0].task is not created[1].task


def test_run_agent_preserves_failed_sample_workspaces(tmp_path):
    module = _load_module()
    created = []

    class Workspace:
        def __init__(self, task, root):
            self.task = task
            self.root = root

        def close(self):
            pass

    def empty_workspace(item):
        workspace = Workspace(dict(item.record), tmp_path / f"sample-{item.sample_index}")
        workspace.root.mkdir()
        created.append(workspace)
        return workspace

    async def fake_run_item(ctx, item, workspace):
        del ctx, workspace
        if item.sample_index == 1:
            raise RuntimeError("no assistant message_end events")
        return [f"turn-{item.sample_index}"]

    items = [
        types.SimpleNamespace(record={"id": "task-1"}, prompt_index=0, sample_index=0),
        types.SimpleNamespace(record={"id": "task-1"}, prompt_index=0, sample_index=1),
    ]
    batch = types.SimpleNamespace(iter_samples=lambda: iter(items))
    ctx = types.SimpleNamespace(max_running_prompts=2)
    module.CodingWorkspace = Workspace
    module._empty_workspace = empty_workspace
    module._run_item = fake_run_item

    trajectory = asyncio.run(module.run_agent(ctx, batch))

    assert trajectory.turns == ["turn-0"]
    assert trajectory.invalid_items == [items[1]]
    assert created[0].root.exists()
    assert created[1].root.exists()


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
