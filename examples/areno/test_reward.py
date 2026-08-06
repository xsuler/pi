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


def _svg(index: int) -> str:
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">'
            f'<circle cx="{64 + index * 20}" cy="256" r="40"/></svg>')


def test_extract_files_accepts_absolute_frame_paths():
    reward = _load_reward_module()
    record = types.SimpleNamespace(tool_calls=[
        {"name": "write", "arguments": {"path": f"/tmp/areno-coding-a1/frame-{i:02d}.svg", "content": _svg(i)}}
        for i in range(8)
    ])
    assert tuple(sorted(reward._extract_files(record))) == reward.REQUIRED_FILES


def test_extract_files_uses_last_write_and_rejects_nested_relative_paths():
    reward = _load_reward_module()
    record = types.SimpleNamespace(tool_calls=[
        {"name": "write", "arguments": {"path": "frame-00.svg", "content": "first"}},
        {"name": "write", "arguments": {"path": "frame-00.svg", "content": "final"}},
        {"name": "write", "arguments": {"path": "nested/frame-01.svg", "content": "wrong"}},
    ])
    assert reward._extract_files(record) == {"frame-00.svg": "final"}


def test_reward_reads_workspace_and_cleans_it(monkeypatch):
    reward = _load_reward_module()
    workspace = Path(tempfile.mkdtemp(prefix="areno-coding-"))
    for i, name in enumerate(reward.REQUIRED_FILES):
        (workspace / name).write_text(_svg(i), encoding="utf-8")
    record = types.SimpleNamespace(source_record={"id": "task-1", "_pi_workspace": str(workspace)},
                                   prompt="Animate", trace=[types.SimpleNamespace(type="request")], tool_calls=[])
    monkeypatch.setattr(reward, "_render_svg_frames", lambda files, sample_id: [b"\x89PNG\r\n\x1a\nframe"] * 8)
    monkeypatch.setattr(reward, "_judge", lambda *args: (8.0, 8.0, 8.0, 8.0))
    assert reward.reward_fn(record) > 0
    assert not workspace.exists()


def test_judge_sends_all_png_frames_in_order(monkeypatch):
    reward = _load_reward_module()
    monkeypatch.setenv("PI_ARENO_JUDGE_BASE_URL", "http://judge.test/v1")
    monkeypatch.setenv("PI_ARENO_JUDGE_API_KEY", "key")
    monkeypatch.setenv("PI_ARENO_JUDGE_MODEL", "vision")
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self):
            scores = {name: [7] * count for name, count in reward.RUBRIC_COUNTS.items()}
            return json.dumps({"choices": [{"message": {"content": json.dumps(scores)}}]}).encode()

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    pngs = [b"\x89PNG\r\n\x1a\n" + bytes([i]) for i in range(8)]
    reward._judge(pngs, {name: _svg(i) for i, name in enumerate(reward.REQUIRED_FILES)}, "brief")
    content = captured["messages"][0]["content"]
    assert [block["text"] for block in content if block["type"] == "text"][1:] == [f"Frame {i:02d} of 07" for i in range(8)]
    assert len([block for block in content if block["type"] == "image_url"]) == 8


def test_rubric_mean_requires_every_score():
    reward = _load_reward_module()
    assert reward._rubric_mean({"svg_quality": [7] * 8}, "svg_quality") == 7
    try:
        reward._rubric_mean({"svg_quality": [7]}, "svg_quality")
    except ValueError as exc:
        assert "exactly 8" in str(exc)
    else:
        raise AssertionError("incomplete rubric accepted")


def test_judge_score_calibration_is_discriminative():
    reward = _load_reward_module()
    assert reward._calibrate_judge_scores(5, 5, 5, 5) == 0
    assert reward._calibrate_judge_scores(8, 8, 8, 8) == 0.6
    assert reward._calibrate_judge_scores(3, 3, 3, 3) == -0.4


def test_validate_svg_rejects_external_resources():
    reward = _load_reward_module()
    reward._validate_svg(_svg(0), "frame-00.svg")
    bad = _svg(0).replace("</svg>", '<image href="https://x/y.png"/></svg>')
    try:
        reward._validate_svg(bad, "frame-00.svg")
    except ValueError:
        pass
    else:
        raise AssertionError("external SVG resource accepted")
