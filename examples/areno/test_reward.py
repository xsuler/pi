from __future__ import annotations

import importlib.util
import json
import tempfile
import types
import urllib.request
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
        source_record={"id": "task-1", "_pi_workspace": str(workspace)},
        prompt="Build a page",
        trace=[types.SimpleNamespace(type="request")],
        tool_calls=[],
    )
    captured = {}

    def fake_render(files, sample_id):
        captured.update(files)
        assert sample_id == "task-1"
        return b"\x89PNG\r\n\x1a\nrendered"

    monkeypatch.setattr(reward, "_html_to_svg", fake_render)
    monkeypatch.setattr(reward, "_judge", lambda *args: (8.0, 8.0, 8.0, 8.0))

    score = reward.reward_fn(record)

    assert score > 0
    assert captured["index.html"] == "<main>final</main>"
    assert not workspace.exists()


def test_judge_sends_png_data_url(monkeypatch):
    reward = _load_reward_module()
    monkeypatch.setenv("PI_ARENO_JUDGE_BASE_URL", "http://judge.test/v1")
    monkeypatch.setenv("PI_ARENO_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("PI_ARENO_JUDGE_MODEL", "vision-model")
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps({
                "html_quality": 8,
                "functional_completeness": 7,
                "visual_alignment": 6,
                "visual_aesthetics": 5,
            })}}]}).encode()

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    files = {name: "content" for name in reward.REQUIRED_FILES}
    reward._judge(b"\x89PNG\r\n\x1a\nimage", files, "brief")

    image_url = captured["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
