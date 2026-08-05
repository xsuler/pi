"""Vision-model reward for Pi-generated HTML, CSS, and JavaScript pages."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

REQUIRED_FILES = ("index.html", "styles.css", "app.js")
logger = logging.getLogger(__name__)


def reward_fn(record) -> float:
    workspace = _record_workspace(record)
    try:
        files = _extract_workspace_files(workspace) if workspace is not None else _extract_files(record)
        if set(files) != set(REQUIRED_FILES):
            logger.warning("Pi web reward missing required files: found=%s workspace=%s", sorted(files), workspace)
            return -1.0
        rendered_svg = _html_to_svg(files, _sample_id(record))
        html_quality, functionality, alignment, aesthetics = _judge(
            rendered_svg,
            files,
            str(record.source_record.get("design_prompt") or record.prompt),
        )
    except Exception as exc:
        logger.warning("Pi web reward evaluation failed: %s", exc, exc_info=True)
        return -1.0
    finally:
        if workspace is not None:
            shutil.rmtree(workspace, ignore_errors=True)
    turns = max(sum(event.type == "request" for event in record.trace), 1)
    limit = max(int(record.source_record.get("max_turns") or 8), 1)
    efficiency = max(0.0, min(1.0, (limit - turns + 1) / limit))
    judged_score = (
        0.20 * html_quality
        + 0.25 * functionality
        + 0.30 * alignment
        + 0.25 * aesthetics
    ) / 10.0
    return max(-1.0, min(1.0, 0.9 * judged_score + 0.1 * efficiency))


def _record_workspace(record) -> Path | None:
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
    files: dict[str, str] = {}
    for name in REQUIRED_FILES:
        path = workspace / name
        if not path.is_file() or path.stat().st_size > 1_000_000:
            continue
        files[name] = path.read_text(encoding="utf-8")
    return files


def _extract_files(record) -> dict[str, str]:
    files: dict[str, str] = {}
    for call in record.tool_calls:
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
        path = str(args.get("path") or "").replace("\\", "/")
        parsed_path = PurePosixPath(path)
        name = parsed_path.name
        content = args.get("content")
        is_direct_path = parsed_path.is_absolute() or parsed_path.parent == PurePosixPath(".")
        if name in REQUIRED_FILES and is_direct_path and isinstance(content, str) and len(content) <= 1_000_000:
            files[name] = content
    return files


def _html_to_svg(files: dict[str, str], sample_id: str = "sample") -> bytes:
    with tempfile.TemporaryDirectory(prefix="pi-web-reward-") as directory:
        root = Path(directory)
        for name, content in files.items():
            (root / name).write_text(content, encoding="utf-8")
        screenshot = root / "page.png"
        profile = root / "chromium-profile"
        command = [
            _chromium_binary(),
            "--headless=new",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-default-apps",
            "--disable-dev-shm-usage",
            "--disable-extensions",
            "--disable-features=Translate,MediaRouter",
            "--disable-gpu",
            "--hide-scrollbars",
            "--host-resolver-rules=MAP * ~NOTFOUND",
            "--no-first-run",
            "--no-sandbox",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=1000",
            "--window-size=1440,1024",
            f"--user-data-dir={profile}",
            f"--screenshot={screenshot}",
            (root / "index.html").as_uri(),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        if completed.returncode != 0 or not screenshot.is_file():
            details = (completed.stderr or completed.stdout)[-4000:]
            raise RuntimeError(f"Chromium rendering failed with exit code {completed.returncode}: {details}")
        png = screenshot.read_bytes()
    encoded = base64.b64encode(png).decode("ascii")
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="1024" viewBox="0 0 1440 1024">'
        f'<image width="1440" height="1024" href="data:image/png;base64,{encoded}"/>'
        "</svg>"
    ).encode("utf-8")
    _save_image_sample(sample_id, png, svg)
    return svg


def _sample_id(record: Any) -> str:
    source = getattr(record, "source_record", {})
    raw = source.get("id") if isinstance(source, dict) else None
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(raw or "sample")).strip("-.") or "sample"


def _save_image_sample(sample_id: str, png: bytes, svg: bytes) -> None:
    output = Path("/tmp/areno_html")
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{sample_id}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    (output / f"{stem}.png").write_bytes(png)
    (output / f"{stem}.svg").write_bytes(svg)


def _chromium_binary() -> str:
    configured = os.environ.get("PI_ARENO_CHROMIUM")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise RuntimeError(f"PI_ARENO_CHROMIUM does not exist: {path}")
        return str(path.resolve())
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError("web-page reward requires Chromium; set PI_ARENO_CHROMIUM to its executable")


def _judge(svg: bytes, files: dict[str, str], prompt: str) -> tuple[float, float, float, float]:
    base_url = _required_env("PI_ARENO_JUDGE_BASE_URL").rstrip("/")
    api_key = _required_env("PI_ARENO_JUDGE_API_KEY")
    model = _required_env("PI_ARENO_JUDGE_MODEL")
    endpoint = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
    image_url = f"data:image/svg+xml;base64,{base64.b64encode(svg).decode('ascii')}"
    source = "\n\n".join(
        f"--- {name} ---\n{files[name][:200_000]}" for name in REQUIRED_FILES
    )
    rubric = (
        "Evaluate a generated web implementation using two distinct evidence tracks. Return JSON only with numeric "
        'fields "html_quality", "functional_completeness", "visual_alignment", and "visual_aesthetics", each from '
        "0 to 10.\n\n"
        "HTML QUALITY PRINCIPLES (judge the supplied HTML/CSS/JS source, not the screenshot): valid and meaningful "
        "semantic structure; coherent responsive CSS; accessible labels, focus states, contrast intent, and keyboard "
        "affordances; functional JavaScript interactions; realistic states and content; maintainability; no external "
        "network dependencies, embedded raster assets, unsafe scripts, or implementation narration.\n\n"
        "FUNCTIONAL COMPLETENESS PRINCIPLES (judge source against the brief): every requested interaction must have "
        "real event handling and visible state feedback; controls must not be decorative shells; navigation, filters, "
        "dialogs, forms, validation, toggles, and state transitions requested by the brief must form usable end-to-end "
        "flows; loading, empty, error, and success states should be represented where relevant.\n\n"
        "VISUAL ALIGNMENT PRINCIPLES (judge the SVG image against the brief): required information architecture, "
        "content, controls, page density, domain specificity, and interaction affordances must be visibly represented.\n\n"
        "VISUAL AESTHETICS PRINCIPLES (judge only the SVG image): hierarchy, typography, spacing, balance, contrast, "
        "color coherence, polish, and absence of overlap, clipping, generic template styling, or excessive decoration. "
        "The image is the rendered 1440x1024 first viewport.\n\n"
        f"DESIGN BRIEF:\n{prompt}\n\nSOURCE FILES:\n{source}"
    )
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": rubric},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]}],
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    match = re.search(r"\{[\s\S]*\}", content)
    scores = json.loads(match.group(0) if match else content)
    return (
        _bounded(scores.get("html_quality")),
        _bounded(scores.get("functional_completeness")),
        _bounded(scores.get("visual_alignment")),
        _bounded(scores.get("visual_aesthetics")),
    )


def _bounded(value: Any) -> float:
    return max(0.0, min(10.0, float(value)))


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value
