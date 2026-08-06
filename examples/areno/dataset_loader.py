"""Dataset loader for 4096 independently specified SVG animation tasks."""

from __future__ import annotations

from typing import Any

EXPECTED_TASKS = 4096


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row_index, row in enumerate(default_loader(dataset_path), start=1):
        record = dict(row)
        task_id = str(record.get("id") or "").strip()
        prompt = str(record.get("design_prompt") or record.get("prompt") or "").strip()
        if not task_id:
            raise ValueError(f"animation task row {row_index} is missing id")
        if task_id in seen_ids:
            raise ValueError(f"duplicate web task id: {task_id}")
        if not prompt:
            raise ValueError(f"animation task {task_id} is missing design_prompt")
        seen_ids.add(task_id)
        record["id"] = task_id
        record["prompt"] = prompt
        record["design_prompt"] = prompt
        record.pop("max_turns", None)
        record["files"] = {"README.md": "Create frame-00.svg through frame-07.svg as a seamless 8 FPS loop.\n"}
        records.append(record)
    if len(records) != EXPECTED_TASKS:
        raise ValueError(f"expected {EXPECTED_TASKS} animation tasks, found {len(records)}")
    return records
