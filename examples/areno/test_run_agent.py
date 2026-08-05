from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


class _Turn:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


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


def _event(message):
    return json.dumps({"type": "message_end", "message": message})


def test_events_skip_non_trainable_assistant_messages():
    module = _load_module()
    lines = [
        _event({"role": "assistant", "content": [{"type": "text", "text": "internal"}]}),
        _event(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "answer"}],
                "providerMetadata": {
                    "areno": {"input_tokens": [1], "response_tokens": [2], "response_logprobs": [-0.2]}
                },
            }
        ),
    ]

    turns = module._events_to_turns(object(), lines)

    assert len(turns) == 1
    assert turns[0].response_tokens == [2]


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
