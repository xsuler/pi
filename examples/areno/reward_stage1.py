"""Deterministic Stage 1 reward for basic HTML, CSS, and JavaScript generation."""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any

REQUIRED_FILES = ("index.html", "styles.css", "app.js")
logger = logging.getLogger(__name__)


class _DocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: set[str] = set()
        self.has_doctype = False
        self.stylesheet_linked = False
        self.script_linked = False

    def handle_decl(self, decl: str) -> None:
        self.has_doctype = decl.strip().lower() == "doctype html"

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.add(tag.lower())
        values = {key.lower(): value or "" for key, value in attrs}
        if tag.lower() == "link" and values.get("rel", "").lower() == "stylesheet":
            self.stylesheet_linked |= PurePosixPath(values.get("href", "")).name == "styles.css"
        if tag.lower() == "script":
            self.script_linked |= PurePosixPath(values.get("src", "")).name == "app.js"


def reward_fn(record) -> float:
    workspace = _record_workspace(record)
    try:
        files = _extract_workspace_files(workspace) if workspace is not None else _extract_files(record)
        if not files:
            return -1.0
        return _score(files)
    except Exception as exc:
        logger.warning("Pi Stage 1 reward failed: %s", exc)
        return -1.0
    finally:
        if workspace is not None:
            shutil.rmtree(workspace, ignore_errors=True)


def _score(files: dict[str, str]) -> float:
    score = 0.0
    if set(files) == set(REQUIRED_FILES) and all(files[name].strip() for name in REQUIRED_FILES):
        score += 0.20
    else:
        score += 0.05 * sum(bool(files.get(name, "").strip()) for name in REQUIRED_FILES)

    html = files.get("index.html", "")
    css = files.get("styles.css", "")
    javascript = files.get("app.js", "")
    parser = _DocumentParser()
    parser.feed(html)
    if parser.has_doctype and {"html", "head", "body", "title"}.issubset(parser.tags):
        score += 0.20
    if any(tag in parser.tags for tag in ("button", "input", "select")):
        score += 0.05
    if parser.stylesheet_linked and css.strip():
        score += 0.075
    if parser.script_linked and javascript.strip():
        score += 0.075
    if _valid_css(css):
        score += 0.15
    if _valid_javascript(javascript):
        score += 0.20
    if not _contains_external_resource(html):
        score += 0.05
    return round(max(-1.0, min(1.0, score)), 6)


def _valid_css(css: str) -> bool:
    stripped = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL).strip()
    return bool(stripped and "{" in stripped and ":" in stripped and stripped.count("{") == stripped.count("}"))


def _valid_javascript(source: str) -> bool:
    stripped = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.DOTALL).strip()
    has_event = "addEventListener" in stripped or re.search(r"\.on(?:click|change|input)\s*=", stripped)
    return bool(
        stripped
        and has_event
        and stripped.count("{") == stripped.count("}")
        and stripped.count("(") == stripped.count(")")
    )


def _contains_external_resource(html: str) -> bool:
    return bool(re.search(r"(?:src|href)\s*=\s*['\"]\s*(?:https?:)?//", html, flags=re.IGNORECASE))


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
    files = {}
    for name in REQUIRED_FILES:
        path = workspace / name
        if path.is_file() and path.stat().st_size <= 1_000_000:
            files[name] = path.read_text(encoding="utf-8")
    return files


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
