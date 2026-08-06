from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).with_name("reward_stage1.py")
    spec = importlib.util.spec_from_file_location("pi_areno_reward_stage1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_files() -> dict[str, str]:
    return {
        "index.html": (
            '<!doctype html><html><head><title>Stock desk</title><link rel="stylesheet" href="styles.css">'
            '</head><body><h1>Stock desk</h1><button id="update">Update</button><p id="status"></p>'
            '<script src="app.js"></script></body></html>'
        ),
        "styles.css": "body { color: #123; } button:focus { outline: 2px solid blue; }",
        "app.js": "document.querySelector('#update').addEventListener('click', () => { status.textContent='Ready'; });",
    }


def test_complete_basic_page_receives_full_score():
    reward = _load_module()

    assert reward._score(_valid_files()) == 1.0


def test_missing_file_and_javascript_interaction_reduce_score():
    reward = _load_module()
    files = _valid_files()
    files.pop("app.js")

    assert reward._score(files) < 0.7


def test_external_resource_loses_self_containment_credit():
    reward = _load_module()
    files = _valid_files()
    files["index.html"] = files["index.html"].replace("styles.css", "https://example.com/styles.css")

    assert reward._score(files) < 1.0
