"""Deterministic Stage 1 reward for an 8-frame SVG animation."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any

REQUIRED_FILES = tuple(f"frame-{index:02d}.svg" for index in range(8))
VISIBLE_ELEMENTS = {"circle", "ellipse", "line", "path", "polygon", "polyline", "rect", "text"}
logger = logging.getLogger(__name__)


def reward_fn(record) -> float:
    workspace = _record_workspace(record)
    try:
        files = _extract_workspace_files(workspace) if workspace is not None else _extract_files(record)
        return _score(files)
    except Exception as exc:
        logger.warning("Pi Stage 1 animation reward failed: %s", exc)
        return -1.0
    finally:
        if workspace is not None:
            shutil.rmtree(workspace, ignore_errors=True)


def _score(files: dict[str, str]) -> float:
    present = [name for name in REQUIRED_FILES if files.get(name, "").strip()]
    if len(present) != 8:
        return round(-1.0 + len(present) / 8.0, 6)
    valid = 0
    visible = 0
    safe = 0
    hashes = set()
    for name in REQUIRED_FILES:
        source = files[name]
        hashes.add(hashlib.sha256(re.sub(r"\s+", " ", source).encode()).digest())
        try:
            root = ET.fromstring(source)
        except ET.ParseError:
            continue
        if root.tag.rsplit("}", 1)[-1].lower() != "svg":
            continue
        view_box = re.split(r"[ ,]+", root.attrib.get("viewBox", "").strip())
        if view_box == ["0", "0", "512", "512"]:
            valid += 1
        tags = {element.tag.rsplit("}", 1)[-1].lower() for element in root.iter()}
        visible += bool(tags & VISIBLE_ELEMENTS)
        lowered = source.lower()
        forbidden = {"animate", "animatemotion", "animatetransform", "foreignobject", "image", "script", "set"}
        unsafe_reference = re.search(r"(?:href|src)\s*=\s*['\"]\s*(?:data:|file:|https?:|//)", lowered)
        safe += not (tags & forbidden or unsafe_reference or "url(" in lowered)
    score = 0.35 * valid / 8 + 0.20 * visible / 8 + 0.20 * safe / 8
    score += 0.25 * min(max((len(hashes) - 1) / 6, 0), 1)
    return round(max(-1.0, min(1.0, score)), 6)


def _record_workspace(record: Any) -> Path | None:
    source = getattr(record, "source_record", None)
    raw = source.get("_pi_workspace") if isinstance(source, dict) else None
    if not raw:
        return None
    workspace = Path(str(raw)).resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()
    if workspace.parent != temp_root or not workspace.name.startswith("areno-coding-"):
        raise RuntimeError(f"refusing unsafe Pi workspace path: {workspace}")
    return workspace


def _extract_workspace_files(workspace: Path) -> dict[str, str]:
    return {
        name: (workspace / name).read_text(encoding="utf-8")
        for name in REQUIRED_FILES
        if (workspace / name).is_file() and (workspace / name).stat().st_size <= 1_000_000
    }


def _extract_files(record: Any) -> dict[str, str]:
    files = {}
    for call in getattr(record, "tool_calls", []):
        if call.get("name") != "write":
            continue
        args = call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if not isinstance(args, dict):
            continue
        name = PurePosixPath(str(args.get("path") or "").replace("\\", "/")).name
        content = args.get("content")
        if name in REQUIRED_FILES and isinstance(content, str) and len(content) <= 1_000_000:
            files[name] = content
    return files
