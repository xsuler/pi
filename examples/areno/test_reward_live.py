#!/usr/bin/env python3
"""Run the real Pi SVG-animation reward judge against one workspace."""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
from pathlib import Path

import reward


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        required=True,
        help="Directory containing frame-00.svg through frame-07.svg",
    )
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="Design brief passed to the judge")
    prompt.add_argument("--prompt-file", help="UTF-8 file containing the design brief")
    parser.add_argument("--sample-id", default="live-reward-test")
    return parser.parse_args()


def _design_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    return Path(args.prompt_file).expanduser().read_text(encoding="utf-8")


def _print_configuration(workspace: Path, files: dict[str, str]) -> None:
    base_url = os.environ.get("PI_ARENO_JUDGE_BASE_URL", "<unset>").rstrip("/")
    endpoint = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
    print(f"workspace: {workspace}")
    print(f"endpoint: {endpoint}")
    print(f"model: {os.environ.get('PI_ARENO_JUDGE_MODEL', '<unset>')}")
    for name in reward.REQUIRED_FILES:
        print(f"file {name}: {len(files.get(name, '').encode('utf-8'))} bytes")


def main() -> int:
    args = _parse_args()
    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.is_dir():
        raise RuntimeError(f"workspace does not exist: {workspace}")

    files = reward._extract_workspace_files(workspace)
    missing = sorted(set(reward.REQUIRED_FILES) - set(files))
    if missing:
        raise RuntimeError(f"workspace is missing required files: {', '.join(missing)}")
    _print_configuration(workspace, files)

    print("rendering eight SVG frames to 512x512 PNGs...")
    pngs = reward._render_svg_frames(files, args.sample_id)
    print(f"rendered PNG bytes: {sum(map(len, pngs))}")
    print("saved frames under: /tmp/areno_animation")
    print("calling multimodal judge...")
    try:
        scores = reward._judge(pngs, files, _design_prompt(args))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code} {exc.reason}", file=sys.stderr)
        print("response headers:", file=sys.stderr)
        for key, value in exc.headers.items():
            print(f"  {key}: {value}", file=sys.stderr)
        print("response body:", file=sys.stderr)
        print(body or "<empty>", file=sys.stderr)
        return 2
    except urllib.error.URLError as exc:
        print(f"request failed before an HTTP response: {exc.reason!r}", file=sys.stderr)
        return 3

    names = ("svg_quality", "motion_continuity", "prompt_alignment", "visual_aesthetics")
    for name, value in zip(names, scores, strict=True):
        print(f"{name}: {value:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
