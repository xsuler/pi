from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path


def _load_reward_module():
    path = Path(__file__).with_name("reward.py")
    spec = importlib.util.spec_from_file_location("pi_areno_reward", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_extract_files_accepts_absolute_isolated_workspace_paths():
    reward = _load_reward_module()
    record = types.SimpleNamespace(
        tool_calls=[
            {
                "name": "write",
                "arguments": json.dumps(
                    {"path": "/tmp/areno-coding-a1/index.html", "content": "<main>ok</main>"}
                ),
            },
            {
                "name": "write",
                "arguments": {"path": "/tmp/areno-coding-a1/styles.css", "content": "main { display: block; }"},
            },
            {
                "name": "write",
                "arguments": {"path": "/tmp/areno-coding-a1/app.js", "content": "console.log('ok');"},
            },
        ]
    )

    assert set(reward._extract_files(record)) == {"index.html", "styles.css", "app.js"}


def test_extract_files_uses_last_write_and_rejects_nested_relative_paths():
    reward = _load_reward_module()
    record = types.SimpleNamespace(
        tool_calls=[
            {"name": "write", "arguments": {"path": "index.html", "content": "first"}},
            {"name": "write", "arguments": {"path": "index.html", "content": "final"}},
            {"name": "write", "arguments": {"path": "nested/styles.css", "content": "wrong location"}},
        ]
    )

    assert reward._extract_files(record) == {"index.html": "final"}
