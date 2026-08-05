from __future__ import annotations

import importlib.util
import json
import tempfile
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


def test_reward_reads_final_workspace_and_cleans_it(monkeypatch):
    reward = _load_reward_module()
    workspace = Path(tempfile.mkdtemp(prefix="areno-coding-"))
    (workspace / "index.html").write_text("<main>final</main>", encoding="utf-8")
    (workspace / "styles.css").write_text("main { display: block; }", encoding="utf-8")
    (workspace / "app.js").write_text("console.log('final');", encoding="utf-8")
    record = types.SimpleNamespace(
        source_record={"id": "task-1", "max_turns": 8, "_pi_workspace": str(workspace)},
        prompt="Build a page",
        trace=[types.SimpleNamespace(type="request")],
        tool_calls=[],
    )
    captured = {}

    def fake_render(files, sample_id):
        captured.update(files)
        assert sample_id == "task-1"
        return b"svg"

    monkeypatch.setattr(reward, "_html_to_svg", fake_render)
    monkeypatch.setattr(reward, "_judge", lambda *args: (8.0, 8.0, 8.0, 8.0))

    score = reward.reward_fn(record)

    assert score > 0
    assert captured["index.html"] == "<main>final</main>"
    assert not workspace.exists()
