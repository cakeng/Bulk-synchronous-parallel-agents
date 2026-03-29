"""run_agent() — structured LLM call with automatic output enforcement and retries.

Usage inside an operator:
    from src.run_agent import run_agent

    parsed, raw, thinking, tool_calls, tokens = await run_agent(
        user_input    = "What is 2 + 2?",
        output_config = {"answer": int, "explanation": str},
        agent_state   = _local,
    )

run_agent automatically:
  1. Creates a Context from ``agent_config["context"]`` (the message list).
  2. Injects a default system prompt when the context is empty.
  3. Appends ``user_input`` as a user message.
  4. Sends the full context to the model.
  5. Appends the model reply as an assistant message and writes the updated
     context back to ``agent_config["context"]`` (only on success).

Optional keys in ``agent_config`` (``agent_state["agent_config"]``):
    context       (list) — existing messages; default [].
    system_prompt (str)  — overrides the default system prompt.
    max_retries   (int)  — how many correction attempts; default 3.
    Any other keys are forwarded to ``chat.completions.create()``.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin

from openai import AsyncOpenAI

from .context import Context


_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# Keys in agent_config never forwarded to chat.completions.create()
_RESERVED = frozenset({
    "base_url", "api_key", "model", "max_retries",
    "context", "system_prompt", "tools", "workspace_dir", "call_log",
    "timeout",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_agent(
    user_input:    str,
    output_config: Optional[Dict[str, type]],
    agent_state:   Dict[str, Any],
    max_retries:   int = 3,
) -> Tuple[Any, str, List[str], List[Tuple], Dict[str, int]]:
    """Execute one LLM call with optional structured output enforcement.

    Args:
        user_input:    The user message to send to the model.
        output_config: Mapping of output key → Python type.  The function
                       appends a JSON-enforcement prompt and retries on parse
                       failure.  Pass ``None`` to skip enforcement and return
                       the raw model string.
        agent_state:   The full ``_local`` agent state dict.  Must contain
                       ``agent_config`` with ``base_url``, ``api_key``, and
                       ``model``.  Optional keys in ``agent_config``:
                       ``context`` (list) — existing messages;
                       ``system_prompt`` (str) — injected when context is
                       empty; ``max_retries`` (int) overrides the parameter.
                       Any remaining keys are forwarded to
                       ``chat.completions.create()``.
        max_retries:   How many correction attempts before raising.

    Returns:
        (parsed_output, raw_model_output, raw_thinking_output,
         tool_calls_and_outputs, tokens_used)

        parsed_output        – dict matching output_config (or the raw string
                               if output_config is None).
        raw_model_output     – full content string from the API response,
                               including any <think> tags.
        raw_thinking_output  – list of thinking strings extracted from
                               <think>/<thinking> tags or a separate
                               reasoning_content field.
        tool_calls_and_outputs – list of (tool_call_object, tool_output)
                               tuples.  tool_output is None until tool
                               execution is wired in.
        tokens_used          – {"total", "prompt", "generation",
                               "reasoning", "tool_calls"}.
    """
    agent_config  = agent_state["agent_config"]
    workspace_dir = agent_state.get("workspace_dir")
    agent_rank    = int(agent_state.get("agent_rank", 0))
    llm_timeout   = float(agent_config.get("timeout", 300.0))
    client = AsyncOpenAI(
        base_url=agent_config["base_url"],
        api_key=agent_config["api_key"],
        timeout=agent_config.get("timeout", 300.0),
    )
    model       = agent_config["model"]
    max_retries = agent_config.get("max_retries", max_retries)
    gen_kwargs  = {k: v for k, v in agent_config.items() if k not in _RESERVED}

    # Pass tool schemas to the API if the agent has tools configured
    tool_schemas = agent_config.get("tools")
    if tool_schemas:
        gen_kwargs["tools"] = tool_schemas

    # Build context from the stored message list; inject system prompt if empty
    ctx = Context(agent_config.get("context", []))
    if not ctx.messages:
        if workspace_dir:
            from src import tools as _tools
            system_prompt = _tools.build_system_prompt(workspace_dir)
        else:
            system_prompt = agent_config.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)
        ctx.system(system_prompt)

    tokens_used: Dict[str, int] = {
        "total": 0, "prompt": 0, "generation": 0,
        "reasoning": 0, "tool_calls": 0,
    }

    # Append the user turn — done once regardless of retries
    ctx.user(user_input)

    # ------------------------------------------------------------------
    # No structured output — single call (with tool loop), return raw
    # ------------------------------------------------------------------
    if output_config is None:
        raw, thinking, tool_calls, extra = await _agentic_loop(
            client, model, ctx.messages, gen_kwargs, workspace_dir, tokens_used,
            agent_rank=agent_rank, llm_timeout=llm_timeout,
        )
        ctx.messages.extend(extra)
        ctx.assistant(raw)
        agent_config["context"] = ctx.messages
        agent_config.setdefault("call_log", []).append({
            "thinking":   thinking,
            "tool_calls": [_serialize_tool_call(tc) for tc, _ in tool_calls],
            "tokens":     dict(tokens_used),
        })
        return (raw, raw, thinking, tool_calls, tokens_used)

    # ------------------------------------------------------------------
    # Structured output — enforce JSON, retry on failure
    # ------------------------------------------------------------------
    enforcement_msg = {"role": "user", "content": _build_json_prompt(output_config)}
    working_context = list(ctx.messages) + [enforcement_msg]

    last_raw   = ""
    last_error = ""
    last_extra: List[Dict] = []

    for attempt in range(max_retries + 1):
        if attempt > 0:
            # Append the bad reply and the correction request
            working_context.append({"role": "assistant", "content": last_raw})
            working_context.append({
                "role":    "user",
                "content": (
                    f"Error: {last_error}. "
                    "Please respond with only a corrected JSON object."
                ),
            })

        raw, thinking, tool_calls, extra = await _agentic_loop(
            client, model, working_context, gen_kwargs, workspace_dir, tokens_used,
            agent_rank=agent_rank, llm_timeout=llm_timeout,
        )
        # Accumulate tool intermediaries into working_context for the next retry
        working_context.extend(extra)
        last_raw   = raw
        last_extra = extra

        try:
            parsed = _parse_and_cast(raw, output_config)
            # Save clean context: original messages + tool intermediaries + final reply
            ctx.messages.extend(last_extra)
            ctx.assistant(raw)
            agent_config["context"] = ctx.messages
            agent_config.setdefault("call_log", []).append({
                "thinking":   thinking,
                "tool_calls": [_serialize_tool_call(tc) for tc, _ in tool_calls],
                "tokens":     dict(tokens_used),
            })
            return (parsed, raw, thinking, tool_calls, tokens_used)
        except ValueError as exc:
            last_error = str(exc)

    raise RuntimeError(
        f"run_agent: failed to obtain valid structured output after "
        f"{max_retries + 1} attempt(s). Last error: {last_error}\n"
        f"Last raw output: {last_raw!r}"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _emit_status(status: str) -> None:
    """Print a structured status event to stdout for the engine to relay to the UI."""
    import sys
    print(json.dumps({"type": "agent_status", "status": status}), flush=True, file=sys.stdout)


def _emit_log(text: str) -> None:
    """Print a log line to stdout so it appears in the agent log dialog."""
    import sys
    print(json.dumps({"type": "agent_log", "stream": "stdout", "text": text}), flush=True, file=sys.stdout)


async def _request_tool_slot(agent_rank: int) -> None:
    """Ask the engine for a tool-execution slot and suspend until granted.

    Uses asyncio.to_thread so the worker's event loop remains free while
    waiting — this is critical: blocking the event loop here would prevent
    httpx from processing TCP events (keep-alive FINs from vLLM) and could
    cause the next LLM call to hang on a silently-dead connection.

    Only active when stdin is a pipe (i.e. running inside an engine subprocess).
    Standalone / direct invocations (stdin is a tty) skip the handshake.
    """
    import sys
    if sys.stdin.isatty():
        return
    print(json.dumps({"type": "tool_slot_request", "agent_rank": agent_rank}), flush=True, file=sys.stdout)
    await asyncio.to_thread(sys.stdin.readline)  # suspends coroutine, not the event loop


def _release_tool_slot(agent_rank: int) -> None:
    """Notify the engine that the tool-execution slot is no longer needed."""
    import sys
    if sys.stdin.isatty():
        return
    print(json.dumps({"type": "tool_slot_release", "agent_rank": agent_rank}), flush=True, file=sys.stdout)


def _truncate(obj) -> str:
    s = json.dumps(obj, ensure_ascii=False) if not isinstance(obj, str) else obj
    return s


async def _agentic_loop(
    client,
    model: str,
    messages: List[Dict],
    gen_kwargs: dict,
    workspace_dir: Optional[str],
    tokens_used: Dict[str, int],
    agent_rank: int = 0,
    llm_timeout: float = 300.0,
) -> Tuple[str, List[str], List[Tuple], List[Dict]]:
    """Call the model in a loop, executing tool calls until a text response arrives.

    Returns:
        (raw, thinking, all_tool_calls, extra_messages)

        raw            – final text content from the model.
        thinking       – aggregated thinking strings across all iterations.
        all_tool_calls – list of (tool_call_object, tool_output) pairs.
        extra_messages – tool-call assistant messages + tool-result messages
                         appended during the loop (NOT including the original
                         ``messages``).  Callers splice these into their own
                         context before saving.
    """
    extra: List[Dict] = []
    all_tool_calls: List[Tuple] = []
    all_thinking: List[str] = []
    current = list(messages)

    if workspace_dir:
        from src import tools as _tools_mod
    else:
        _tools_mod = None  # type: ignore[assignment]

    call_index = 0
    while True:
        call_index += 1
        _emit_status("Waiting for LLM")
        _emit_log(f"[LLM call #{call_index}] sending {len(current)} message(s)…")
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(model=model, messages=current, **gen_kwargs),
                timeout=llm_timeout,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"[LLM call #{call_index}] no response after {llm_timeout:.0f}s — "
                "vLLM may have dropped the request (check context length / server state)"
            )
        message = response.choices[0].message
        raw, thinking = _extract_content(message)
        tool_calls    = _extract_tool_calls(message)
        _accumulate_tokens(tokens_used, response.usage)
        all_thinking.extend(thinking)
        all_tool_calls.extend(tool_calls)

        usage = response.usage
        tok_info = (
            f"prompt={getattr(usage,'prompt_tokens',0)}  "
            f"completion={getattr(usage,'completion_tokens',0)}"
            if usage else ""
        )
        if tok_info:
            _emit_log(f"[LLM call #{call_index}] tokens: {tok_info}")

        if not tool_calls:
            if raw:
                _emit_log(f"[LLM call #{call_index}] response: {_truncate(raw)}")
            return raw, all_thinking, all_tool_calls, extra

        _emit_log(f"[LLM call #{call_index}] {len(tool_calls)} tool call(s) requested")

        # ── Execute tool calls and continue the loop ──────────────────────
        tc_dicts = [_serialize_tool_call(tc) for tc, _ in tool_calls]
        asst_msg: Dict = {
            "role":       "assistant",
            "content":    raw or None,
            "tool_calls": tc_dicts,
        }
        current.append(asst_msg)
        extra.append(asst_msg)

        for tc, _ in tool_calls:
            fn = getattr(tc, "function", None)
            fn_name     = getattr(fn, "name",      "") or ""
            fn_args_str = getattr(fn, "arguments", "{}") or "{}"
            try:
                fn_args = json.loads(fn_args_str)
            except (json.JSONDecodeError, TypeError):
                fn_args = {}

            _emit_status(f"Tool: {fn_name}")
            _emit_log(f"[tool] {fn_name}({_truncate(fn_args)})")
            if _tools_mod is not None:
                await _request_tool_slot(agent_rank)
                try:
                    # Run the (blocking) tool dispatch in a thread so the event
                    # loop stays free to process network events while the tool runs.
                    result = await asyncio.to_thread(
                        _tools_mod.dispatch_function_call, fn_name, fn_args, workspace_dir
                    )
                finally:
                    _release_tool_slot(agent_rank)
            else:
                result = {"error": "No workspace configured for tool execution"}

            _emit_log(f"[tool result] {_truncate(result)}")

            tool_msg: Dict = {
                "role":         "tool",
                "tool_call_id": getattr(tc, "id", ""),
                "content":      json.dumps(result),
            }
            current.append(tool_msg)
            extra.append(tool_msg)


_THINK_RE = re.compile(
    r"<think(?:ing)?>(.*?)</think(?:ing)?>",
    re.DOTALL | re.IGNORECASE,
)


def _extract_content(message) -> Tuple[str, List[str]]:
    """Return (raw_content_string, list_of_thinking_strings).

    Thinking is collected from:
    1. A ``reasoning_content`` attribute on the message (Qwen3 / DeepSeek via
       vLLM's ``--reasoning-parser``).
    2. ``<think>`` / ``<thinking>`` tags inside the content string.
    """
    thinking: List[str] = []

    # Source 1: separate reasoning field (vLLM uses "reasoning" or "reasoning_content"
    # depending on the parser version / model; check both)
    rc = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    if rc:
        thinking.append(rc)

    content = message.content or ""

    # Source 2: inline tags
    for m in _THINK_RE.finditer(content):
        thinking.append(m.group(1).strip())

    return content, thinking


def _extract_tool_calls(message) -> List[Tuple]:
    """Return list of (tool_call_object, None) — outputs filled in later."""
    if not getattr(message, "tool_calls", None):
        return []
    return [(tc, None) for tc in message.tool_calls]


def _accumulate_tokens(acc: Dict[str, int], usage) -> None:
    if usage is None:
        return
    acc["total"]  += getattr(usage, "total_tokens",      0) or 0
    acc["prompt"] += getattr(usage, "prompt_tokens",     0) or 0
    completion     = getattr(usage, "completion_tokens", 0) or 0

    details   = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", 0) or 0 if details else 0

    acc["reasoning"]  += reasoning
    acc["generation"] += completion - reasoning


def _serialize_tool_call(tc) -> dict:
    """Serialize an OpenAI tool_call object to a plain JSON-safe dict."""
    fn = getattr(tc, "function", None)
    return {
        "id":   getattr(tc, "id",   ""),
        "type": getattr(tc, "type", "function"),
        "function": {
            "name":      getattr(fn, "name",      "") if fn else "",
            "arguments": getattr(fn, "arguments", "") if fn else "",
        },
    }


_TYPE_NAMES = {
    str: "string", int: "integer", float: "float",
    bool: "boolean", list: "list", dict: "object", tuple: "tuple",
}


def _type_description(t) -> str:
    """Return a human-readable description of a type, including generic parameters."""
    origin = get_origin(t)
    args   = get_args(t)
    if origin is list:
        inner = _type_description(args[0]) if args else "value"
        # simple pluralisation: append 's' unless already ends in 's'
        plural = inner if inner.endswith("s") else inner + "s"
        return f"list of {plural}"
    if origin is dict:
        if len(args) == 2:
            return f"object with {_type_description(args[0])} keys and {_type_description(args[1])} values"
        return "object"
    if origin is tuple:
        if args:
            return "tuple of (" + ", ".join(_type_description(a) for a in args) + ")"
        return "tuple"
    return _TYPE_NAMES.get(t, getattr(t, "__name__", str(t)))


def _type_placeholder(t) -> str:
    """Return a JSON-value placeholder like <STRING> or [<INTEGER>, ...] for a type."""
    origin = get_origin(t)
    args   = get_args(t)
    if origin is list:
        inner = _type_placeholder(args[0]) if args else "<VALUE>"
        return f"[{inner}, ...]"
    if origin is dict:
        k_ph = _type_placeholder(args[0]) if args else "<KEY>"
        v_ph = _type_placeholder(args[1]) if len(args) > 1 else "<VALUE>"
        return "{" + f"{k_ph}: {v_ph}, ..." + "}"
    if origin is tuple:
        return "[" + ", ".join(_type_placeholder(a) for a in args) + "]" if args else "[...]"
    name = _TYPE_NAMES.get(t, getattr(t, "__name__", str(t)))
    return f"<{name.upper()}>"


def _build_json_prompt(output_config: Dict[str, type]) -> str:
    desc_lines = "\n".join(
        f'  "{k}": {_type_description(dtype)}'
        for k, dtype in output_config.items()
    )
    example_fields = "\n".join(
        f'  "{k}": {_type_placeholder(dtype)}'
        for k, dtype in output_config.items()
    )
    return (
        "Your response must be a valid JSON object and nothing else — "
        "no markdown fences, no explanation, no text before or after the JSON.\n"
        f"Required keys and their expected types:\n{desc_lines}\n\n"
        f"Example format:\n{{\n{example_fields}\n}}"
    )


def _parse_and_cast(raw: str, output_config: Dict[str, type]) -> Dict[str, Any]:
    """Parse JSON from raw string and cast each value to the configured type.

    Raises ValueError with user-facing messages on any failure so the caller
    can append them to the chat context for self-correction.
    """
    text = raw.strip()

    # Strip thinking tags before JSON parsing
    text = _THINK_RE.sub("", text).strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    if text.startswith("```"):
        lines = text.splitlines()
        end   = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text  = "\n".join(lines[1:end]).strip()

    # Extract the first syntactically valid JSON object from the text.
    # Using raw_decode in all cases handles both preamble text (stray "{" chars
    # like "{20} papers") and trailing text after the closing "}" correctly.
    _decoder = json.JSONDecoder()
    _found = False
    for _i, _c in enumerate(text):
        if _c == "{":
            try:
                data, _ = _decoder.raw_decode(text, _i)
                _found = True
                break
            except json.JSONDecodeError:
                pass
    if not _found:
        raise ValueError("Output not JSON")

    if not isinstance(data, dict):
        raise ValueError("Output not JSON")

    result: Dict[str, Any] = {}
    for key, dtype in output_config.items():
        if key not in data:
            raise ValueError(f"Key {key} missing")
        raw_val = data[key]
        try:
            # Special-case bool: Python's bool("false") == True, which is wrong
            if dtype is bool:
                if isinstance(raw_val, bool):
                    result[key] = raw_val
                elif isinstance(raw_val, str) and raw_val.lower() in ("true", "false"):
                    result[key] = raw_val.lower() == "true"
                else:
                    result[key] = bool(raw_val)
            else:
                result[key] = dtype(raw_val)
        except (TypeError, ValueError):
            raise ValueError(f"{key} value not {dtype.__name__}")

    return result
