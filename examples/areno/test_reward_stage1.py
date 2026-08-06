from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("reward_stage1.py")
    spec = importlib.util.spec_from_file_location("pi_areno_reward_stage1", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _frames(unique: bool = True):
    return {
        f"frame-{index:02d}.svg": (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">'
            f'<circle cx="{64 + (index if unique else 0) * 20}" cy="256" r="40" fill="#e34"/></svg>'
        )
        for index in range(8)
    }


def test_complete_animation_receives_full_score():
    assert _load_module()._score(_frames()) == 1.0


def test_missing_frames_are_penalized():
    files = _frames()
    files.pop("frame-07.svg")
    assert _load_module()._score(files) < 0


def test_duplicate_frames_are_penalized():
    assert _load_module()._score(_frames(unique=False)) < 0.8


def test_external_resources_are_penalized():
    files = _frames()
    files["frame-03.svg"] = files["frame-03.svg"].replace("</svg>", '<image href="https://x/y.png"/></svg>')
    assert _load_module()._score(files) < 1.0
