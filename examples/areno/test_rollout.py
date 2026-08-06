#!/usr/bin/env python3
"""Run one real Pi rollout against an AReno OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = """Create exactly three complete files in the current directory: index.html, styles.css, and app.js.
Build a polished responsive task tracker with an input, add button, completion toggles, delete controls, and persisted
state using localStorage. Do not load external resources. Use Pi's tools to inspect the directory and write the files,
then give a concise final answer."""
ALL_PI_TOOLS = "read,bash,edit,write,grep,find,ls"


def main() -> int:
    args = _parse_args()
    binary = Path(args.binary).expanduser().resolve()
    if not binary.is_file():
        raise SystemExit(f"Pi binary not found: {binary}")

    workspace_context = (
        tempfile.TemporaryDirectory(prefix="pi-areno-rollout-") if args.workspace is None else _ExistingDirectory()
    )
    with workspace_context as temporary_workspace, tempfile.TemporaryDirectory(prefix="pi-areno-config-") as agent_dir:
        workspace = Path(args.workspace).expanduser().resolve() if args.workspace else Path(temporary_workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        _write_models(Path(agent_dir), args.base_url, args.api_key)
        command = [
            str(binary),
            "--mode",
            "json",
            "--print",
            "--no-session",
            "--no-extensions",
            "--no-skills",
            "--no-context-files",
            "--offline",
            "--tools",
            ALL_PI_TOOLS,
            "--provider",
            "areno",
            "--model",
            "policy",
            "--api-key",
            args.api_key,
            args.prompt,
        ]
        result = subprocess.run(
            command,
            cwd=workspace,
            env={
                **os.environ,
                "PI_CODING_AGENT_DIR": agent_dir,
                "PI_OFFLINE": "1",
                "PI_MAX_TURNS": "20",
            },
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
        events = _parse_events(result.stdout.splitlines())
        if args.raw:
            print(result.stdout, end="")
        trainable, tool_calls = _print_summary(events, workspace, result)
        if result.returncode != 0 or trainable == 0:
            return 1
        if args.require_tool_call and tool_calls == 0:
            print("FAIL: rollout produced no Pi tool calls", file=sys.stderr)
            return 2
        return 0


class _ExistingDirectory:
    def __enter__(self) -> str:
        return ""

    def __exit__(self, *exc_info: object) -> None:
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("ARENO_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("ARENO_API_KEY", "areno-agentic"))
    parser.add_argument("--binary", default=os.environ.get("PI_ARENO_BINARY", "./pi"))
    parser.add_argument("--workspace", help="Use and modify this directory instead of a temporary workspace")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--raw", action="store_true", help="Print the complete Pi JSONL stream before the summary")
    parser.add_argument("--require-tool-call", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not args.base_url:
        parser.error("--base-url or ARENO_BASE_URL is required")
    return args


def _write_models(agent_dir: Path, base_url: str, api_key: str) -> None:
    config = {
        "providers": {
            "areno": {
                "baseUrl": base_url,
                "api": "openai-completions",
                "apiKey": api_key,
                "compat": {
                    "supportsStreaming": False,
                    "supportsStore": False,
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                    "supportsUsageInStreaming": False,
                    "supportsStrictMode": False,
                    "maxTokensField": "max_tokens",
                },
                "models": [
                    {
                        "id": "policy",
                        "name": "AReno Policy",
                        "contextWindow": 128000,
                        "maxTokens": 16384,
                        "samplingParams": {"temperature": 1.0},
                    }
                ],
            }
        }
    }
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "models.json").write_text(json.dumps(config), encoding="utf-8")


def _parse_events(lines: list[str]) -> list[dict[str, Any]]:
    events = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def _print_summary(events: list[dict[str, Any]], workspace: Path, result: subprocess.CompletedProcess[str]) -> tuple[int, int]:
    trainable = 0
    tool_calls = 0
    print(f"Pi exit code: {result.returncode}")
    print(f"Workspace: {workspace}")
    for event in events:
        if event.get("type") != "message_end" or not isinstance(event.get("message"), dict):
            continue
        message = event["message"]
        role = message.get("role")
        if role == "assistant":
            metadata = message.get("providerMetadata", {}).get("areno", {})
            response_tokens = metadata.get("response_tokens")
            content = message.get("content")
            content_parts = content if isinstance(content, list) else []
            calls = [part for part in content_parts if isinstance(part, dict) and part.get("type") == "toolCall"]
            tool_calls += len(calls)
            if isinstance(response_tokens, list):
                trainable += 1
            print(
                f"assistant stop={message.get('stopReason')!r} input_tokens={len(metadata.get('input_tokens') or [])} "
                f"response_tokens={len(response_tokens or [])} tool_calls={len(calls)}"
            )
            if message.get("errorMessage"):
                print(f"  error: {message['errorMessage']}")
            text = "".join(
                str(part.get("text") or "")
                for part in content_parts
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()
            if isinstance(content, str):
                text = content.strip()
            if text:
                print(f"  text: {text[:500]}")
            for call in calls:
                print(f"  tool: {call.get('name')} {json.dumps(call.get('arguments'), ensure_ascii=False)[:500]}")
        elif role == "toolResult":
            print(f"tool_result name={message.get('toolName')!r} error={message.get('isError')!r}")

    print(f"Summary: trainable_turns={trainable} tool_calls={tool_calls} events={len(events)}")
    for name in ("index.html", "styles.css", "app.js"):
        path = workspace / name
        print(f"file {name}: {'present' if path.is_file() else 'missing'} size={path.stat().st_size if path.is_file() else 0}")
    if result.stderr.strip():
        print("Pi stderr:", file=sys.stderr)
        print(result.stderr[-4000:], file=sys.stderr)
    return trainable, tool_calls


if __name__ == "__main__":
    raise SystemExit(main())
