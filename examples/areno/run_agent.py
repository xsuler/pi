"""Run the real Pi coding-agent loop as an AReno trainable agent."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from areno.agent.tools import CodingWorkspace
from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

logger = logging.getLogger(__name__)
ALL_PI_TOOLS = "read,bash,edit,write,grep,find,ls"


async def run_agent(ctx, batch) -> AgentTrajectory:
    items = list(batch.iter_samples())
    for item in items:
        item.record = dict(item.record)
    semaphore = asyncio.Semaphore(ctx.max_running_prompts)

    async def run_one(item):
        async with semaphore:
            workspace = None
            try:
                workspace = await asyncio.to_thread(_empty_workspace, item)
                return await _run_item(ctx, item, workspace)
            except BaseException:
                if workspace is not None:
                    _discard_workspace(workspace)
                raise
            finally:
                if workspace is not None:
                    workspace.close()

    results = await asyncio.gather(
        *(run_one(item) for item in items),
        return_exceptions=True,
    )
    valid_turns = []
    invalid_items = []
    for item, result in zip(items, results, strict=True):
        if isinstance(result, BaseException):
            invalid_items.append(item)
            logger.warning(
                "filtering invalid Pi rollout prompt_index=%s sample_index=%s error=%s",
                item.prompt_index,
                item.sample_index,
                result,
            )
        else:
            valid_turns.extend(result)
    return AgentTrajectory(turns=valid_turns, invalid_items=invalid_items)


async def _run_item(ctx, item, workspace: CodingWorkspace) -> list[AgentTrajectoryTurn]:
    with tempfile.TemporaryDirectory(prefix="pi-areno-config-") as agent_dir:
        _write_models(Path(agent_dir), ctx.get_base_url(), ctx.api_key)
        lines = await _run_pi_process(ctx, workspace.root, agent_dir, _prompt(item))
        return _events_to_turns(item, lines)


def _empty_workspace(item) -> CodingWorkspace:
    root = Path(tempfile.mkdtemp(prefix="areno-coding-"))
    item.record["_pi_workspace"] = str(root)
    return CodingWorkspace(task=item.record, root=root, cleanup_on_close=False)


def _discard_workspace(workspace: CodingWorkspace) -> None:
    shutil.rmtree(workspace.root, ignore_errors=True)


async def _run_pi_process(ctx, workspace: Path, agent_dir: str, prompt: str) -> list[str]:
    process = await asyncio.create_subprocess_exec(
        str(_pi_binary()),
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
        ctx.api_key,
        prompt,
        cwd=workspace,
        env={**os.environ, "PI_CODING_AGENT_DIR": agent_dir, "PI_OFFLINE": "1"},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=900)
    except BaseException:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        raise
    if process.returncode != 0:
        raise RuntimeError(f"Pi exited with {process.returncode}: {stderr.decode(errors='replace')[-4000:]}")
    return stdout.decode(errors="replace").splitlines()


def _events_to_turns(item, lines: list[str]) -> list[AgentTrajectoryTurn]:
    messages: list[dict[str, Any]] = []
    turns: list[AgentTrajectoryTurn] = []
    assistant_diagnostics: list[str] = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "message_end" or not isinstance(event.get("message"), dict):
            continue
        message = event["message"]
        if message.get("role") == "assistant":
            metadata = _areno_metadata(message)
            tokens = metadata.get("response_tokens")
            logprobs = metadata.get("response_logprobs")
            input_tokens = metadata.get("input_tokens")
            if not all(isinstance(value, list) for value in (tokens, logprobs, input_tokens)):
                assistant_diagnostics.append(
                    f"keys={sorted(message)} stopReason={message.get('stopReason')!r} "
                    f"errorMessage={message.get('errorMessage')!r} "
                    f"providerMetadata={message.get('providerMetadata')!r}"
                )
            else:
                assistant_message = _openai_message(message) or {"role": "assistant", "content": ""}
                tool_calls = assistant_message.get("tool_calls") or []
                turns.append(
                    AgentTrajectoryTurn(
                        item=item,
                        messages=list(messages),
                        response={
                            "choices": [
                                {
                                    "index": 0,
                                    "message": assistant_message,
                                    "finish_reason": "tool_calls" if tool_calls else "stop",
                                }
                            ],
                            "areno": {
                                "input_tokens": [int(token) for token in input_tokens],
                                "response_tokens": [int(token) for token in tokens],
                                "response_logprobs": [float(value) for value in logprobs],
                            },
                        },
                    )
                )
        normalized = _openai_message(message)
        if normalized is not None:
            messages.append(normalized)
    if not turns:
        details = "; ".join(assistant_diagnostics[-3:]) or "no assistant message_end events"
        raise RuntimeError(f"Pi produced no trainable assistant turns with AReno metadata: {details}")
    return turns


def _areno_metadata(message: dict[str, Any]) -> dict[str, Any]:
    """Read AReno metadata across Pi's typed and wire-compatible representations."""
    candidates = (
        message.get("providerMetadata"),
        message.get("provider_metadata"),
        message.get("metadata"),
    )
    for candidate in candidates:
        if isinstance(candidate, dict) and isinstance(candidate.get("areno"), dict):
            return candidate["areno"]
    return message.get("areno") if isinstance(message.get("areno"), dict) else {}


def _openai_message(message: dict[str, Any]) -> dict[str, Any] | None:
    role = message.get("role")
    content = message.get("content")
    if role == "user":
        return {"role": "user", "content": _text(content)}
    if role == "toolResult":
        return {
            "role": "tool",
            "tool_call_id": message.get("toolCallId"),
            "name": message.get("toolName"),
            "content": _text(content),
        }
    if role == "assistant":
        calls = _tool_calls(message)
        result: dict[str, Any] = {"role": "assistant", "content": _text(content) or None}
        if calls:
            result["tool_calls"] = calls
        return result
    return None


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for block in message.get("content") or []:
        if block.get("type") != "toolCall":
            continue
        calls.append(
            {
                "id": str(block.get("id") or ""),
                "type": "function",
                "function": {
                    "name": str(block.get("name") or ""),
                    "arguments": json.dumps(block.get("arguments") or {}, separators=(",", ":")),
                },
            }
        )
    return calls


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(str(block.get("text") or "") for block in content if block.get("type") == "text")


def _prompt(item) -> str:
    return (
        f"{item.prompt}\nCreate exactly three complete files in the current directory: index.html, styles.css, and "
        "app.js. Build a polished responsive implementation of the design brief with all requested interactions and "
        "persisted state where relevant. Do not load external resources. Use Pi's tools to inspect the directory and "
        "write the files, then give a concise final answer."
    )


def _write_models(agent_dir: Path, base_url: str, api_key: str) -> None:
    agent_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "providers": {
            "areno": {
                "baseUrl": base_url,
                "api": "openai-completions",
                "apiKey": api_key,
                "compat": {
                    "supportsStreaming": False,
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                    "maxTokensField": "max_tokens",
                },
                "models": [{"id": "policy", "name": "AReno Policy"}],
            }
        }
    }
    (agent_dir / "models.json").write_text(json.dumps(config), encoding="utf-8")


def _pi_binary() -> Path:
    configured = os.environ.get("PI_ARENO_BINARY")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "packages" / "coding-agent" / "dist" / "pi"
