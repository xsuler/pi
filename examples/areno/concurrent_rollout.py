#!/usr/bin/env python3
"""Run concurrent real Pi rollouts against an active AReno agentic proxy."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALL_PI_TOOLS = "read,bash,edit,write,grep,find,ls"
REQUIRED_FILES = ("index.html", "styles.css", "app.js")
TOOL_PROTOCOL_PROMPT = """When creating the requested page, follow this tool protocol exactly:
- The working directory already exists. Never create it and never pass the working-directory path to write.
- Call write once per turn with a relative path that is exactly index.html, styles.css, or app.js.
- Include non-empty content in every write call. Keep each file compact enough to fit within 700 output tokens.
- Never run rm, rm -rf, or rmdir on the workspace or any created file. Leave all three required files in place.
- Finish all three files before replying with final text. If a tool fails, correct its arguments instead of repeating them."""


@dataclass(slots=True)
class Result:
    prompt_index: int
    sample_index: int
    workspace: Path
    event_log: Path
    returncode: int
    elapsed_s: float
    trainable_turns: int
    tool_calls: int
    tool_results: int
    tool_errors: int
    file_sizes: dict[str, int]
    last_text: str
    stderr: str

    @property
    def valid(self) -> bool:
        return (
            self.returncode == 0
            and self.trainable_turns > 0
            and self.tool_calls > 0
            and all(self.file_sizes.get(name, 0) > 0 for name in REQUIRED_FILES)
        )


async def main_async(args: argparse.Namespace) -> int:
    binary = Path(args.binary).expanduser().resolve()
    if not binary.is_file():
        raise SystemExit(f"Pi binary not found: {binary}")
    prompts = _load_prompts(Path(args.dataset_path), args.records)
    run_root = Path(args.workspace_root).expanduser().resolve() / (
        time.strftime("%Y%m%d-%H%M%S") + f"-{uuid.uuid4().hex[:6]}"
    )
    run_root.mkdir(parents=True)
    log_root = run_root / "logs"
    log_root.mkdir()
    jobs = [
        (prompt_index, sample_index, prompt)
        for prompt_index, prompt in enumerate(prompts)
        for sample_index in range(args.n_samples)
    ]
    concurrency = args.max_concurrency or len(jobs)
    semaphore = asyncio.Semaphore(concurrency)
    print(
        f"run_root={run_root} jobs={len(jobs)} prompts={len(prompts)} "
        f"n_samples={args.n_samples} concurrency={concurrency}"
    )

    async def run_job(prompt_index: int, sample_index: int, prompt: str) -> Result:
        async with semaphore:
            return await _run_one(args, binary, run_root, log_root, prompt_index, sample_index, prompt)

    results = await asyncio.gather(*(run_job(*job) for job in jobs))
    for result in results:
        sizes = ",".join(f"{name}={result.file_sizes[name]}" for name in REQUIRED_FILES)
        print(
            f"[{result.prompt_index}:{result.sample_index}] valid={result.valid} rc={result.returncode} "
            f"elapsed={result.elapsed_s:.2f}s turns={result.trainable_turns} calls={result.tool_calls} "
            f"results={result.tool_results} tool_errors={result.tool_errors} files=({sizes}) "
            f"workspace={result.workspace} events={result.event_log}"
        )
        if not result.valid:
            if result.last_text:
                print(f"  last_text={result.last_text[:500]!r}")
            if result.stderr:
                print(f"  stderr={result.stderr[-1000:]!r}")

    valid = sum(result.valid for result in results)
    with_files = sum(all(result.file_sizes[name] > 0 for name in REQUIRED_FILES) for result in results)
    with_tools = sum(result.tool_calls > 0 for result in results)
    with_tokens = sum(result.trainable_turns > 0 for result in results)
    print(
        f"SUMMARY valid={valid}/{len(results)} with_tokens={with_tokens}/{len(results)} "
        f"with_tools={with_tools}/{len(results)} with_all_files={with_files}/{len(results)} run_root={run_root}"
    )
    return 0 if valid == len(results) else 1


async def _run_one(
    args: argparse.Namespace,
    binary: Path,
    run_root: Path,
    log_root: Path,
    prompt_index: int,
    sample_index: int,
    prompt: str,
) -> Result:
    workspace = run_root / f"prompt-{prompt_index:04d}-sample-{sample_index:02d}"
    workspace.mkdir()
    log_prefix = log_root / f"prompt-{prompt_index:04d}-sample-{sample_index:02d}"
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="pi-areno-config-") as agent_dir:
        _write_models(Path(agent_dir), args.base_url, args.api_key)
        process = await asyncio.create_subprocess_exec(
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
            "--append-system-prompt",
            TOOL_PROTOCOL_PROMPT,
            "--provider",
            "areno",
            "--model",
            "policy",
            "--api-key",
            args.api_key,
            _training_prompt(prompt),
            cwd=workspace,
            env={**os.environ, "PI_CODING_AGENT_DIR": agent_dir, "PI_OFFLINE": "1"},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=args.timeout)
        except asyncio.TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()
            stderr += b"\nconcurrent rollout timed out"
    events = _parse_events(stdout.decode(errors="replace").splitlines())
    summary = _summarize(events)
    sizes = _generated_file_sizes(workspace)
    event_log = log_prefix.with_suffix(".events.jsonl")
    event_log.write_bytes(stdout)
    log_prefix.with_suffix(".stderr.log").write_bytes(stderr)
    return Result(
        prompt_index=prompt_index,
        sample_index=sample_index,
        workspace=workspace,
        event_log=event_log,
        returncode=int(process.returncode or 0),
        elapsed_s=time.monotonic() - started,
        trainable_turns=summary["trainable_turns"],
        tool_calls=summary["tool_calls"],
        tool_results=summary["tool_results"],
        tool_errors=summary["tool_errors"],
        file_sizes=sizes,
        last_text=summary["last_text"],
        stderr=stderr.decode(errors="replace").strip(),
    )


def _summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    result = {"trainable_turns": 0, "tool_calls": 0, "tool_results": 0, "tool_errors": 0, "last_text": ""}
    for event in events:
        if event.get("type") != "message_end" or not isinstance(event.get("message"), dict):
            continue
        message = event["message"]
        if message.get("role") == "assistant":
            metadata = (message.get("providerMetadata") or {}).get("areno") or {}
            if isinstance(metadata.get("response_tokens"), list):
                result["trainable_turns"] += 1
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "toolCall":
                    result["tool_calls"] += 1
                elif isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                    result["last_text"] = str(block["text"])
        elif message.get("role") == "toolResult":
            result["tool_results"] += 1
            result["tool_errors"] += bool(message.get("isError"))
    return result


def _generated_file_sizes(workspace: Path) -> dict[str, int]:
    return {
        name: (workspace / name).stat().st_size if (workspace / name).is_file() else 0 for name in REQUIRED_FILES
    }


def _parse_events(lines: list[str]) -> list[dict[str, Any]]:
    events = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _training_prompt(prompt: str) -> str:
    return (
        f"{prompt}\nCreate exactly three complete files in the current directory: index.html, styles.css, and "
        "app.js. Build a polished responsive implementation of the design brief with all requested interactions and "
        "persisted state where relevant. Do not load external resources. Write each required file in a separate tool "
        "turn using only its relative filename, then give a concise final answer."
    )


def _load_prompts(path: Path, count: int) -> list[str]:
    prompts = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            prompts.append(str(record.get("design_prompt") or record.get("prompt") or ""))
            if len(prompts) == count:
                break
    if len(prompts) != count or any(not prompt for prompt in prompts):
        raise RuntimeError(f"dataset supplied {len(prompts)} usable prompts, expected {count}: {path}")
    return prompts


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("ARENO_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("ARENO_API_KEY", "areno-agentic"))
    parser.add_argument("--binary", default=os.environ.get("PI_ARENO_BINARY", "./pi"))
    parser.add_argument(
        "--dataset-path",
        default=str(Path(__file__).with_name("web_tasks_4096_simpler.jsonl")),
    )
    parser.add_argument("--records", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--max-concurrency", type=int, default=0, help="0 means all jobs concurrently")
    parser.add_argument("--workspace-root", default="/tmp/pi-concurrent-rollout")
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    if not args.base_url:
        parser.error("--base-url or ARENO_BASE_URL is required")
    if args.records < 1 or args.n_samples < 1 or args.max_concurrency < 0:
        parser.error("records/n-samples must be positive and max-concurrency must be non-negative")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(_parse_args())))
