from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("concurrent_rollout.py")
    spec = importlib.util.spec_from_file_location("pi_concurrent_rollout", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_prompts_accepts_design_prompt_and_prompt(tmp_path):
    module = _load_module()
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {"design_prompt": "Build the first page"},
                {"prompt": "Build the second page"},
                {"prompt": "This record should not be loaded"},
            )
        ),
        encoding="utf-8",
    )

    assert module._load_prompts(dataset, 2) == ["Build the first page", "Build the second page"]


def test_summarize_counts_trainable_turns_and_tools():
    module = _load_module()
    events = [
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Writing files"},
                    {"type": "toolCall", "name": "write", "arguments": {"path": "index.html"}},
                ],
                "providerMetadata": {"areno": {"response_tokens": [1, 2]}},
            },
        },
        {
            "type": "message_end",
            "message": {"role": "toolResult", "isError": False, "content": [{"type": "text", "text": "ok"}]},
        },
        {
            "type": "message_end",
            "message": {"role": "toolResult", "isError": True, "content": [{"type": "text", "text": "bad"}]},
        },
    ]

    assert module._summarize(events) == {
        "trainable_turns": 1,
        "tool_calls": 1,
        "tool_results": 2,
        "tool_errors": 1,
        "last_text": "Writing files",
    }


def test_parse_events_skips_non_json_and_non_object_lines():
    module = _load_module()

    assert module._parse_events(
        [
            "Pi startup output",
            '{"type":"agent_start"}',
            "[]",
            '{"type":"agent_end"}',
            "{invalid-json",
        ]
    ) == [{"type": "agent_start"}, {"type": "agent_end"}]


def test_training_prompt_requires_eight_frames_and_pi_tools():
    module = _load_module()

    prompt = module._training_prompt("Build a task tracker.")

    assert prompt.startswith("Build a task tracker.")
    assert "frame-00.svg through frame-07.svg" in prompt
    assert "inspect, and edit every frame" in prompt
    assert "separate tool turn" in prompt
    assert "relative filename" in prompt


def test_tool_protocol_prevents_writing_the_workspace_directory():
    module = _load_module()

    assert "Never create it" in module.TOOL_PROTOCOL_PROMPT
    assert "frame-00.svg through frame-07.svg" in module.TOOL_PROTOCOL_PROMPT
    assert "use edit repeatedly" in module.TOOL_PROTOCOL_PROMPT
    assert "Include non-empty content" in module.TOOL_PROTOCOL_PROMPT
    assert "Never run rm, rm -rf, or rmdir" in module.TOOL_PROTOCOL_PROMPT


def test_deleted_workspace_reports_missing_files_without_recreating_it(tmp_path):
    module = _load_module()
    workspace = tmp_path / "deleted-workspace"

    assert module._generated_file_sizes(workspace) == {
        f"frame-{index:02d}.svg": 0 for index in range(8)
    }
    assert not workspace.exists()


def test_models_config_sets_temperature_to_one(tmp_path):
    module = _load_module()

    module._write_models(tmp_path, "http://127.0.0.1:3000/v1", "test-key")

    config = json.loads((tmp_path / "models.json").read_text(encoding="utf-8"))
    provider = config["providers"]["areno"]
    model = config["providers"]["areno"]["models"][0]
    assert provider["compat"] == {
        "supportsStreaming": False,
        "supportsStore": False,
        "supportsDeveloperRole": False,
        "supportsReasoningEffort": False,
        "supportsUsageInStreaming": False,
        "supportsStrictMode": False,
        "maxTokensField": "max_tokens",
    }
    assert model["contextWindow"] == 128000
    assert model["maxTokens"] == 16384
    assert model["samplingParams"] == {"temperature": 1.0}
