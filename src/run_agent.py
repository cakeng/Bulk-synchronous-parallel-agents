"""run_agent() — structured LLM call with automatic output enforcement and retries.

Usage inside an operator:
    from src.run_agent import run_agent

    parsed, raw, thinking, tool_calls, tokens = await run_agent(
        user_input    = "What is 2 + 2?",
        output_config = {"answer": int, "explanation": str},
        agent_config  = _local["llm_state"],
    )

run_agent automatically:
  1. Creates a Context from ``llm_state["context"]`` (the message list).
  2. Injects a default system prompt when the context is empty.
  3. Appends ``user_input`` as a user message.
  4. Sends the full context to the model.
  5. Appends the model reply as an assistant message and writes the updated
     context back to ``llm_state["context"]`` (only on success).

Optional keys in ``llm_state``:
    context       (list) — existing messages; default [].
    system_prompt (str)  — overrides the default system prompt.
    max_retries   (int)  — how many correction attempts; default 3.
    Any other keys are forwarded to ``chat.completions.create()``.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from .context import Context


_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# Keys in agent_config never forwarded to chat.completions.create()
_RESERVED = frozenset({
    "base_url", "api_key", "model", "max_retries",
    "context", "system_prompt",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_agent(
    user_input:    str,
    output_config: Optional[Dict[str, type]],
    agent_config:  Dict[str, Any],
    max_retries:   int = 3,
) -> Tuple[Any, str, List[str], List[Tuple], Dict[str, int]]:
    """Execute one LLM call with optional structured output enforcement.

    Args:
        user_input:    The user message to send to the model.
        output_config: Mapping of output key → Python type.  The function
                       appends a JSON-enforcement prompt and retries on parse
                       failure.  Pass ``None`` to skip enforcement and return
                       the raw model string.
        agent_config:  ``_local["llm_state"]`` — must contain ``base_url``,
                       ``api_key``, and ``model``.  Optional keys:
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
    client = AsyncOpenAI(
        base_url=agent_config["base_url"],
        api_key=agent_config["api_key"],
    )
    model       = agent_config["model"]
    max_retries = agent_config.get("max_retries", max_retries)
    gen_kwargs  = {k: v for k, v in agent_config.items() if k not in _RESERVED}

    # Build context from the stored message list; inject system prompt if empty
    ctx = Context(agent_config.get("context", []))
    if not ctx.messages:
        ctx.system(agent_config.get("system_prompt", _DEFAULT_SYSTEM_PROMPT))

    tokens_used: Dict[str, int] = {
        "total": 0, "prompt": 0, "generation": 0,
        "reasoning": 0, "tool_calls": 0,
    }

    # Append the user turn — done once regardless of retries
    ctx.user(user_input)

    # ------------------------------------------------------------------
    # No structured output — single call, return raw string
    # ------------------------------------------------------------------
    if output_config is None:
        response = await client.chat.completions.create(
            model=model, messages=ctx.messages, **gen_kwargs
        )
        message  = response.choices[0].message
        raw, thinking = _extract_content(message)
        _accumulate_tokens(tokens_used, response.usage)
        ctx.assistant(raw)
        agent_config["context"] = ctx.messages
        return (raw, raw, thinking, _extract_tool_calls(message), tokens_used)

    # ------------------------------------------------------------------
    # Structured output — enforce JSON, retry on failure
    # ------------------------------------------------------------------
    enforcement_msg = {"role": "user", "content": _build_json_prompt(output_config)}
    working_context = list(ctx.messages) + [enforcement_msg]

    last_raw   = ""
    last_error = ""

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

        response = await client.chat.completions.create(
            model=model, messages=working_context, **gen_kwargs
        )
        message  = response.choices[0].message
        raw, thinking = _extract_content(message)
        tool_calls    = _extract_tool_calls(message)
        _accumulate_tokens(tokens_used, response.usage)
        last_raw = raw

        try:
            parsed = _parse_and_cast(raw, output_config)
            ctx.assistant(raw)
            agent_config["context"] = ctx.messages
            return (parsed, raw, thinking, tool_calls, tokens_used)
        except ValueError as exc:
            last_error = str(exc)

    raise RuntimeError(
        f"run_agent: failed to obtain valid structured output after "
        f"{max_retries + 1} attempt(s). Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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

    # Source 1: separate reasoning_content field
    rc = getattr(message, "reasoning_content", None)
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


def _build_json_prompt(output_config: Dict[str, type]) -> str:
    fields = "\n".join(
        f'  "{k}": <{dtype.__name__}>'
        for k, dtype in output_config.items()
    )
    return (
        "Your response must be a valid JSON object and nothing else — "
        "no markdown fences, no explanation, no text before or after the JSON.\n"
        f"The JSON must contain exactly these keys:\n{{\n{fields}\n}}"
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

    # Attempt to extract a JSON object if there is surrounding prose
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
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
