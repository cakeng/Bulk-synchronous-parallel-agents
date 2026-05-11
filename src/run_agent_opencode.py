"""run_agent_opencode() — routes run_agent calls to per-agent opencode subprocesses.

Each agent gets a persistent opencode session (session ID stored in
``agent_state["opencode_session_id"]``) that runs inside the agent's
``workspace_dir``.  Opencode owns context management, auto-compression,
and its full native tool suite (bash, files, web, code, etc.).

Usage inside an operator — exact same call signature as run_agent():

    from src.run_agent_opencode import run_agent_opencode as run_agent

    parsed, raw, thinking, tool_calls, tokens = await run_agent(
        user_input    = "Analyse the repo and summarise findings.",
        output_config = {"summary": str, "key_files": list},
        agent_state   = _local,
    )

Optional keys in ``agent_config``:
    opencode_model  (str)  — passed as --model; default "cefprovider/Qwen/Qwen3.5-35B-A3B-FP8".
    opencode_flags  (list) — extra CLI flags forwarded verbatim to ``opencode run``.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .run_agent import _build_json_prompt, _parse_and_cast


# ---------------------------------------------------------------------------
# Minimal shims so call_log serialisation is identical to run_agent's format
# ---------------------------------------------------------------------------

@dataclass
class _Fn:
    name:      str
    arguments: str  # JSON-encoded string

@dataclass
class _ToolCall:
    id:       str
    type:     str  = "function"
    function: _Fn  = field(default_factory=lambda: _Fn("", "{}"))


def _serialize(tc: _ToolCall) -> dict:
    return {
        "id":   tc.id,
        "type": tc.type,
        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_agent_opencode(
    user_input:    str,
    output_config: Optional[Dict[str, type]],
    agent_state:   Dict[str, Any],
    max_retries:   int = 3,
) -> Tuple[Any, str, List[str], List[Tuple], Dict[str, Any]]:
    """Drop-in replacement for run_agent() backed by an opencode subprocess.

    Key differences from run_agent():
    - No vLLM / OpenAI endpoint required directly; opencode talks to it.
    - Opencode handles context, tool use, and auto-compression natively.
    - The ``opencode`` CLI must be installed and configured (``opencode auth``).
    - Structured-output retries continue the same session (via --session).
    """
    from src import ipc
    from src.ipc import request_gpu_slot, release_gpu_slot

    agent_config  = agent_state.get("agent_config", {})
    workspace_dir = agent_state.get("workspace_dir") or None
    if not workspace_dir or not os.path.isdir(workspace_dir):
        raise RuntimeError(
            f"Agent has no valid workspace_dir (got {workspace_dir!r}). "
            "Ensure the engine initialises workspaces before running operators."
        )
    session_id  = agent_state.get("opencode_session_id")

    if "context" not in agent_config:
        agent_config["context"] = []

    model       = agent_config.get("opencode_model") or "cefprovider/Qwen/Qwen3.5-35B-A3B-FP8"
    extra_flags = list(agent_config.get("opencode_flags", []))

    tokens: Dict[str, Any] = {
        "total": 0, "prompt": 0, "generation": 0,
        "reasoning": 0, "tool_calls": 0, "cost_usd": 0.0,
    }

    prompt   = user_input
    if output_config is not None:
        prompt = user_input + "\n\n" + _build_json_prompt(output_config)

    attempts       = (max_retries + 1) if output_config is not None else 1
    last_raw       = ""
    last_error     = ""
    all_thinking:   List[str]   = []
    all_tool_calls: List[Tuple] = []
    agent_rank   = agent_state.get("agent_rank", 0)
    use_gpu_slot = bool(agent_config.get("use_gpu_slot", True))

    for attempt in range(attempts):
        send_prompt = prompt if attempt == 0 else (
            f"Error: {last_error}. "
            "Please respond with only a corrected JSON object."
        )
        if use_gpu_slot:
            await request_gpu_slot(agent_rank)
        try:
            raw, thinking, tool_calls, new_sid, cost, tok, messages = await _invoke_opencode(
                prompt=send_prompt,
                session_id=session_id,
                workspace_dir=workspace_dir,
                model=model,
                extra_flags=extra_flags,
                ipc=ipc,
            )
        except RuntimeError:
            if session_id:
                ipc.emit({"type": "agent_log", "stream": "stderr",
                          "text": f"[opencode] session {session_id!r} failed — retrying as new session"})
                session_id = None
                agent_state["opencode_session_id"] = None
                agent_config["context"] = []
                raw, thinking, tool_calls, new_sid, cost, tok, messages = await _invoke_opencode(
                    prompt=send_prompt,
                    session_id=None,
                    workspace_dir=workspace_dir,
                    model=model,
                    extra_flags=extra_flags,
                    ipc=ipc,
                )
            else:
                raise
        finally:
            if use_gpu_slot:
                release_gpu_slot(agent_rank)

        if new_sid:
            session_id = new_sid
            agent_state["opencode_session_id"] = new_sid

        tokens["cost_usd"]   += cost
        tokens["total"]      += tok.get("total", 0)
        tokens["prompt"]     += tok.get("input", 0)
        tokens["generation"] += tok.get("output", 0)
        tokens["reasoning"]  += tok.get("reasoning", 0)
        all_thinking.extend(thinking)
        all_tool_calls.extend(tool_calls)
        last_raw = raw

        agent_config["context"].extend(messages)

        if output_config is None:
            _append_call_log(agent_config, all_thinking, all_tool_calls, tokens)
            return (raw, raw, all_thinking, all_tool_calls, tokens)

        try:
            parsed = _parse_and_cast(raw, output_config)
            _append_call_log(agent_config, all_thinking, all_tool_calls, tokens)
            return (parsed, raw, all_thinking, all_tool_calls, tokens)
        except ValueError as exc:
            last_error = str(exc)

    raise RuntimeError(
        f"run_agent_opencode: failed to obtain valid structured output after "
        f"{attempts} attempt(s).  Last error: {last_error}\n"
        f"Last raw output: {last_raw!r}"
    )


def _append_call_log(agent_config, thinking, tool_calls, tokens):
    agent_config.setdefault("call_log", []).append({
        "thinking":   thinking,
        "tool_calls": [_serialize(tc) for tc, _ in tool_calls],
        "tokens":     dict(tokens),
    })


# ---------------------------------------------------------------------------
# Internal: spawn `opencode run` and parse its --format json stream
# ---------------------------------------------------------------------------

async def _invoke_opencode(
    prompt:        str,
    session_id:    Optional[str],
    workspace_dir: str,
    model:         str,
    extra_flags:   List[str],
    ipc,
) -> Tuple[str, List[str], List[Tuple], Optional[str], float, Dict, List[Dict]]:
    """Spawn ``opencode run <prompt> --format json`` and parse output.

    Reads stdout line-by-line in real time so IPC status events are emitted
    during the run, not only at the end.

    Event schema (``--format json`` stream):
      {"type": "<part_type>", "timestamp": ms, "sessionID": "ses_...", "part": {...}}

    Relevant part types:
      text          — assistant text reply       (part.text)
      reasoning     — thinking block             (part.text)
      tool          — tool call / result         (part.callID, part.tool, part.state)
      step_finish   — end of one LLM step        (part.tokens, part.cost)
      error         — error event                (part.message or part.error)

    Returns ``(raw_text, thinking, tool_calls, new_session_id, cost_usd, tokens, messages)``.
    ``messages`` is a reconstructed OpenAI-format conversation list for the chat panel.
    """
    cmd = [
        "opencode", "run", prompt,
        "--format", "json",
        "--dangerously-skip-permissions",
        "--model", model,
    ]
    if session_id:
        cmd += ["--session", session_id]
    cmd.extend(extra_flags)

    env = {**os.environ}
    env["OPENCODE_DISABLE_AUTOUPDATE"] = "true"

    ipc.emit({"type": "agent_status", "status": "Waiting for opencode"})
    ipc.emit({"type": "agent_log", "stream": "stdout",
              "text": f"[opencode] starting (model={model}, session={session_id or 'new'})"})

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=workspace_dir,
        env=env,
    )

    raw_text       = ""
    new_session_id: Optional[str] = session_id
    cost_usd       = 0.0
    agg_tokens: Dict[str, int] = {"total": 0, "input": 0, "output": 0, "reasoning": 0}
    thinking:      List[str]   = []
    tool_calls:    List[Tuple] = []
    stderr_lines:  List[str]   = []
    # Maps callID → index in tool_calls so results can be patched in.
    _tc_index:     Dict[str, int] = {}

    built_messages: List[Dict] = [{"role": "user", "content": prompt}]
    _final_added: List[bool] = [False]

    async def _read_stderr() -> None:
        assert proc.stderr
        async for chunk in proc.stderr:
            line = chunk.decode(errors="replace").rstrip()
            if line:
                stderr_lines.append(line)
                ipc.emit({"type": "agent_log", "stream": "stderr", "text": line})

    async def _read_stdout() -> None:
        nonlocal raw_text, new_session_id, cost_usd
        assert proc.stdout
        async for chunk in proc.stdout:
            line = chunk.decode(errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                ipc.emit({"type": "agent_log", "stream": "stdout", "text": line})
                continue

            # Grab session ID from any event that carries it.
            sid = event.get("sessionID")
            if sid:
                new_session_id = sid

            etype = event.get("type")
            part  = event.get("part") or {}

            # ── Assistant text ────────────────────────────────────────────
            if etype == "text":
                text = part.get("text", "")
                if text:
                    raw_text = text
                    ipc.emit({"type": "agent_log", "stream": "stdout",
                              "text": f"[response] {text[:300]}"})
                    built_messages.append({"role": "assistant", "content": text})
                    _final_added[0] = True

            # ── Thinking / reasoning ──────────────────────────────────────
            elif etype == "reasoning":
                text = part.get("text", "")
                if text:
                    thinking.append(text)
                    ipc.emit({"type": "agent_log", "stream": "stdout",
                              "text": f"[thinking] {text[:300]}"})

            # ── Tool call / result ────────────────────────────────────────
            elif etype == "tool":
                call_id   = part.get("callID", "")
                tool_name = part.get("tool", "")
                state     = part.get("state") or {}
                status    = state.get("status", "")

                if status in ("pending", "running") and call_id not in _tc_index:
                    tc = _ToolCall(
                        id=call_id,
                        function=_Fn(
                            name=tool_name,
                            arguments=json.dumps(state.get("input") or {}),
                        ),
                    )
                    _tc_index[call_id] = len(tool_calls)
                    tool_calls.append((tc, None))
                    ipc.emit({"type": "agent_status", "status": f"Tool: {tool_name}"})
                    ipc.emit({"type": "agent_log", "stream": "stdout",
                              "text": f"[tool] {tool_name}({json.dumps(state.get('input') or {})[:200]})"})
                    built_messages.append({"role": "assistant", "content": ""})

                elif status == "completed":
                    result_text = state.get("output", "")
                    idx = _tc_index.get(call_id)
                    if idx is not None:
                        tc, _ = tool_calls[idx]
                        # Update arguments with final input in case it changed.
                        tc.function.arguments = json.dumps(state.get("input") or {})
                        tool_calls[idx] = (tc, result_text)
                    ipc.emit({"type": "agent_log", "stream": "stdout",
                              "text": f"[tool result] {result_text[:300]}"})
                    built_messages.append({
                        "role":         "tool",
                        "tool_call_id": call_id,
                        "content":      result_text,
                    })

                elif status == "error":
                    error_text = state.get("error", "unknown tool error")
                    idx = _tc_index.get(call_id)
                    if idx is not None:
                        tool_calls[idx] = (tool_calls[idx][0], f"[error] {error_text}")
                    ipc.emit({"type": "agent_log", "stream": "stderr",
                              "text": f"[tool error] {tool_name}: {error_text}"})

            # ── Step finish (tokens + cost) ────────────────────────────────
            elif etype == "step_finish":
                tok  = part.get("tokens") or {}
                cost_usd += float(part.get("cost") or 0.0)
                for k in ("total", "input", "output", "reasoning"):
                    agg_tokens[k] = agg_tokens.get(k, 0) + int(tok.get(k) or 0)

            # ── Error event ────────────────────────────────────────────────
            elif etype == "error":
                msg = part.get("message") or part.get("error") or str(part)
                raise RuntimeError(f"opencode error: {msg}")

    stderr_task = asyncio.create_task(_read_stderr())
    stdout_task = asyncio.create_task(_read_stdout())

    await proc.wait()
    results = await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

    for exc in results:
        if isinstance(exc, Exception):
            raise exc

    if proc.returncode != 0:
        err = "\n".join(stderr_lines).strip() or f"exit code {proc.returncode}"
        raise RuntimeError(f"`opencode` subprocess failed: {err}")

    # Fallback: if no text event fired but we have non-empty raw_text from another
    # path (shouldn't normally happen), still add it to the chat panel.
    if not _final_added[0] and raw_text:
        built_messages.append({"role": "assistant", "content": raw_text})

    ipc.emit({"type": "agent_status", "status": ""})
    ipc.emit({"type": "agent_log", "stream": "stdout",
              "text": f"[opencode] done — cost ${cost_usd:.4f}, session={new_session_id}"})

    return raw_text, thinking, tool_calls, new_session_id, cost_usd, agg_tokens, built_messages
