"""run_agent_claude() — routes run_agent calls to per-agent Claude Code subprocesses.

Each agent gets a persistent Claude Code session (session ID stored in
``agent_state["claude_session_id"]``) that runs inside the agent's
``workspace_dir``.  Claude Code owns context management, auto-compression,
and its full native tool suite (bash, files, web, code, etc.).

Usage inside an operator — exact same call signature as run_agent():

    from src.run_agent_claude import run_agent_claude as run_agent

    parsed, raw, thinking, tool_calls, tokens = await run_agent(
        user_input    = "Analyse the repo and summarise findings.",
        output_config = {"summary": str, "key_files": list},
        agent_state   = _local,
    )

Optional keys in ``agent_config``:
    claude_model  (str)  — passed as --model; default "claude-opus-4-6".
    claude_flags  (list) — extra CLI flags forwarded verbatim to ``claude``.
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
    type:     str   = "function"
    function: _Fn   = field(default_factory=lambda: _Fn("", "{}"))


def _serialize(tc: _ToolCall) -> dict:
    return {
        "id":   tc.id,
        "type": tc.type,
        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_agent_claude(
    user_input:    str,
    output_config: Optional[Dict[str, type]],
    agent_state:   Dict[str, Any],
    max_retries:   int = 3,
) -> Tuple[Any, str, List[str], List[Tuple], Dict[str, Any]]:
    """Drop-in replacement for run_agent() backed by a Claude Code subprocess.

    Key differences from run_agent():
    - No vLLM / OpenAI endpoint required.
    - Claude Code handles context, tool use, and auto-compression natively.
    - The ``claude`` CLI must be installed and authenticated (``claude login``).
    - Structured-output retries continue the same session (via --resume).
    """
    from src import ipc  # imported here to stay subprocess-friendly

    agent_config  = agent_state.get("agent_config", {})
    workspace_dir = agent_state.get("workspace_dir") or None
    if not workspace_dir or not os.path.isdir(workspace_dir):
        raise RuntimeError(
            f"Agent has no valid workspace_dir (got {workspace_dir!r}). "
            "Ensure the engine initialises workspaces before running operators."
        )
    session_id    = agent_state.get("claude_session_id")

    # Preserve existing context across operator calls so the chat panel shows
    # the full conversation history.  New messages are appended below.
    if "context" not in agent_config:
        agent_config["context"] = []
    # Model: claude_model overrides the shared model key so the two harnesses
    # can coexist in the same run with different model selections.
    model       = agent_config.get("claude_model") or agent_config.get("model", "claude-opus-4-6")
    base_url    = agent_config.get("base_url")
    api_key     = agent_config.get("api_key")
    extra_flags = list(agent_config.get("claude_flags", []))

    tokens: Dict[str, Any] = {
        "total": 0, "prompt": 0, "generation": 0,
        "reasoning": 0, "tool_calls": 0, "cost_usd": 0.0,
    }

    # Build first prompt; append JSON schema when structured output is required.
    prompt = user_input
    if output_config is not None:
        prompt = user_input + "\n\n" + _build_json_prompt(output_config)

    attempts       = (max_retries + 1) if output_config is not None else 1
    last_raw       = ""
    last_error     = ""
    all_thinking:   List[str]   = []
    all_tool_calls: List[Tuple] = []

    for attempt in range(attempts):
        send_prompt = prompt if attempt == 0 else (
            f"Error: {last_error}. "
            "Please respond with only a corrected JSON object."
        )

        try:
            raw, thinking, tool_calls, new_sid, cost, messages = await _invoke_claude(
                prompt=send_prompt,
                session_id=session_id,
                workspace_dir=workspace_dir,
                model=model,
                base_url=base_url,
                api_key=api_key,
                extra_flags=extra_flags,
                ipc=ipc,
            )
        except RuntimeError:
            # If we had a session_id and it caused the failure (e.g. stale/expired
            # session from a previous machine), clear it and retry without --resume.
            if session_id:
                ipc.emit({"type": "agent_log", "stream": "stderr",
                          "text": f"[claude] session {session_id!r} failed — retrying as new session"})
                session_id = None
                agent_state["claude_session_id"] = None
                agent_config["context"] = []
                raw, thinking, tool_calls, new_sid, cost, messages = await _invoke_claude(
                    prompt=send_prompt,
                    session_id=None,
                    workspace_dir=workspace_dir,
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    extra_flags=extra_flags,
                    ipc=ipc,
                )
            else:
                raise

        if new_sid:
            session_id = new_sid
            agent_state["claude_session_id"] = new_sid

        tokens["cost_usd"] += cost
        all_thinking.extend(thinking)
        all_tool_calls.extend(tool_calls)
        last_raw = raw

        # Append this attempt's conversation to the context for the chat panel.
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
        f"run_agent_claude: failed to obtain valid structured output after "
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
# Internal: spawn `claude` and parse its stream-json output
# ---------------------------------------------------------------------------

async def _invoke_claude(
    prompt:        str,
    session_id:    Optional[str],
    workspace_dir: str,
    model:         str,
    base_url:      Optional[str],
    api_key:       Optional[str],
    extra_flags:   List[str],
    ipc,
) -> Tuple[str, List[str], List[Tuple], Optional[str], float, List[Dict]]:
    """Spawn ``claude -p <prompt> --output-format stream-json`` and parse output.

    Reads stdout line-by-line in real time so IPC status events are emitted
    during the run, not only at the end.

    ``base_url`` and ``api_key`` are forwarded as ``ANTHROPIC_BASE_URL`` /
    ``ANTHROPIC_API_KEY`` environment variables so the subprocess uses the
    same OpenAI-compatible endpoint as the rest of the engine.

    Returns ``(raw_text, thinking, tool_calls, new_session_id, cost_usd, messages)``.
    ``messages`` is a reconstructed OpenAI-format conversation list suitable for
    storing in ``agent_config["context"]`` for the chat panel.
    """
    cmd = [
        "claude", "-p", prompt,
        "--output-format", "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
        "--model", "opus",  # tier alias → remapped to served model via ANTHROPIC_DEFAULT_OPUS_MODEL
    ]
    if session_id:
        cmd += ["--resume", session_id]
    cmd.extend(extra_flags)

    # Point Claude Code at the vLLM server via Anthropic-compatible env vars.
    #
    # ANTHROPIC_BASE_URL must be the bare host:port — the Anthropic SDK appends
    # /v1/messages itself, so strip any trailing /v1 that agent_config carries.
    #
    # ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL maps Claude tier names to the
    # actual model served by vLLM.  We request "claude-opus-4-5" in --model so
    # Claude Code picks the Opus tier, which vLLM re-routes to our custom model.
    env = {**os.environ}
    if base_url:
        anthropic_base = base_url.rstrip("/")
        if anthropic_base.endswith("/v1"):
            anthropic_base = anthropic_base[:-3]
        env["ANTHROPIC_BASE_URL"] = anthropic_base
    _key = api_key or "dummy"
    env["ANTHROPIC_API_KEY"]   = _key
    env["ANTHROPIC_AUTH_TOKEN"] = _key
    if model:
        env["ANTHROPIC_DEFAULT_OPUS_MODEL"]   = model
        env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"]  = model
    env["API_TIMEOUT_MS"]                        = "6000000"
    env["BASH_DEFAULT_TIMEOUT_MS"]               = "1800000"
    env["BASH_MAX_TIMEOUT_MS"]                   = "7200000"
    env["DISABLE_TELEMETRY"]                     = "1"
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"

    ipc.emit({"type": "agent_status", "status": "Waiting for Claude Code"})
    ipc.emit({"type": "agent_log", "stream": "stdout",
              "text": f"[claude] starting (served_model={model}, base_url={base_url or 'default'}, session={session_id or 'new'})"})

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=workspace_dir,
        env=env,
    )

    raw_text      = ""
    new_session_id: Optional[str] = session_id
    cost_usd      = 0.0
    thinking:      List[str]   = []
    tool_calls:    List[Tuple] = []
    stderr_lines:  List[str]   = []
    # Maps tool_use id → index in tool_calls so results can be patched in.
    _tc_index:     Dict[str, int] = {}

    # Reconstructed OpenAI-format conversation for the chat panel.
    # Starts with the user prompt; assistant/tool messages added as events arrive.
    built_messages: List[Dict] = [{"role": "user", "content": prompt}]
    # Flag (as single-element list to avoid nonlocal for a bool) tracking whether
    # the final text-content assistant message has been appended.
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

            etype = event.get("type")

            # ── Session init ──────────────────────────────────────────────
            if etype == "system" and event.get("subtype") == "init":
                sid = event.get("session_id")
                if sid:
                    new_session_id = sid

            # ── Assistant turn (text / thinking / tool_use blocks) ────────
            elif etype == "assistant":
                msg     = event.get("message", {})
                content = msg.get("content") or []

                # First pass: classify the turn so we know how to build the message.
                _turn_text     = ""
                _turn_has_tool = False
                for block in content:
                    btype = block.get("type")
                    if btype == "text":
                        _turn_text = block.get("text", "")
                    elif btype == "tool_use":
                        _turn_has_tool = True

                # Emit IPC events (second pass keeps existing order).
                for block in content:
                    btype = block.get("type")

                    if btype == "thinking":
                        t = block.get("thinking", "")
                        if t:
                            thinking.append(t)
                            ipc.emit({"type": "agent_log", "stream": "stdout",
                                      "text": f"[thinking] {t[:300]}"})

                    elif btype == "text":
                        raw_text = block.get("text", "")
                        ipc.emit({"type": "agent_log", "stream": "stdout",
                                  "text": f"[response] {raw_text[:300]}"})

                    elif btype == "tool_use":
                        tc_name  = block.get("name", "")
                        tc_input = block.get("input") or {}
                        tc_id    = block.get("id", "")
                        tc = _ToolCall(
                            id=tc_id,
                            function=_Fn(
                                name=tc_name,
                                arguments=json.dumps(tc_input),
                            ),
                        )
                        _tc_index[tc_id] = len(tool_calls)
                        tool_calls.append((tc, None))
                        ipc.emit({"type": "agent_status",
                                  "status": f"Tool: {tc_name}"})
                        ipc.emit({"type": "agent_log", "stream": "stdout",
                                  "text": f"[tool] {tc_name}({json.dumps(tc_input)[:200]})"})

                # Build the context message for the chat panel.
                # Tool-calling turns use empty content so the chat panel's
                # call_log mapper treats them as intermediaries (not the final
                # reply that carries thinking / tool-call annotations).
                if _turn_has_tool:
                    built_messages.append({"role": "assistant", "content": ""})
                elif _turn_text:
                    built_messages.append({"role": "assistant", "content": _turn_text})
                    _final_added[0] = True

            # ── Tool result (user turn carrying tool output) ──────────────
            elif etype == "user":
                msg     = event.get("message", {})
                content = msg.get("content") or []
                for block in content:
                    if block.get("type") == "tool_result":
                        tc_id          = block.get("tool_use_id", "")
                        result_content = block.get("content") or []
                        # content can be a plain string or a list of {"type":"text"} blocks
                        if isinstance(result_content, str):
                            result_text = result_content
                        else:
                            result_text = " ".join(
                                b.get("text", "") for b in result_content
                                if isinstance(b, dict) and b.get("type") == "text"
                            )
                        # Patch the output back into the matching tool call tuple.
                        idx = _tc_index.get(tc_id)
                        if idx is not None:
                            tc, _ = tool_calls[idx]
                            tool_calls[idx] = (tc, result_text)
                        ipc.emit({"type": "agent_log", "stream": "stdout",
                                  "text": f"[tool result] {result_text[:300]}"})
                        # Add to chat panel context.
                        built_messages.append({
                            "role":        "tool",
                            "tool_call_id": tc_id,
                            "content":     result_text,
                        })

            # ── Final result ──────────────────────────────────────────────
            elif etype == "result":
                new_session_id = event.get("session_id", new_session_id)
                cost_usd       = float(event.get("cost_usd") or 0.0)
                if event.get("is_error") or event.get("subtype") == "error":
                    raise RuntimeError(
                        f"Claude Code error: {event.get('result', 'unknown error')}"
                    )
                # result field is the final consolidated text
                final = event.get("result", "")
                if final:
                    raw_text = final
                    # Fallback: if no assistant text-block event fired (some Claude
                    # Code versions only emit text in the result event), add it now.
                    if not _final_added[0]:
                        built_messages.append({"role": "assistant", "content": final})
                        _final_added[0] = True

    stderr_task = asyncio.create_task(_read_stderr())
    stdout_task = asyncio.create_task(_read_stdout())

    await proc.wait()
    await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

    if proc.returncode != 0:
        err = "\n".join(stderr_lines).strip() or f"exit code {proc.returncode}"
        raise RuntimeError(f"`claude` subprocess failed: {err}")

    ipc.emit({"type": "agent_status", "status": ""})
    ipc.emit({"type": "agent_log", "stream": "stdout",
              "text": f"[claude] done — cost ${cost_usd:.4f}, session={new_session_id}"})

    return raw_text, thinking, tool_calls, new_session_id, cost_usd, built_messages
