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


def test_training_prompt_requires_three_files_and_pi_tools():
    module = _load_module()

    prompt = module._training_prompt("Build a task tracker.")

    assert prompt.startswith("Build a task tracker.")
    assert "index.html, styles.css, and app.js" in prompt
    assert "Use Pi's tools" in prompt
