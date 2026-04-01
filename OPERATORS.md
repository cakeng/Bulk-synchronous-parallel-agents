# Operator Authoring Reference

Operators are defined in `runs/<run_name>/operators/`.  The UI editor shows only the **body** of `async def run` — the surrounding class and imports are generated automatically.

---

## The two built-in parameters

### `_local` — agent state dict

Holds every variable that belongs to this agent.  Values written here persist across steps.

```python
# Read a variable set by a previous step
prev = _local["last_reply"]

# Write a variable for the next step to use
_local["score"] = 42
```

The code-generator auto-unpacks plain variable names from `_local` at the top of the function and packs them back in a `finally` block, so you can also write:

```python
score = 42          # same as _local["score"] = 42
prev  = last_reply  # same as prev = _local["last_reply"]
```

Names starting with `_` (e.g. `_parsed`, `_local`, `_global`) are **never** auto-managed.

Always-present keys:

| Key | Type | Description |
|-----|------|-------------|
| `agent_rank` | `int` | 0-based index of this agent in the current run |
| `agent_config` | `dict` | LLM connection settings (`base_url`, `api_key`, `model`, `context`, `call_log`) |
| `workspace_dir` | `str` | Absolute path to this agent's private workspace directory |
| `unique_id` | `str` | Stable UUID for this agent across steps |

---

### `_global` — engine-wide dict

Read-only engine variables shared across all agents in a step.

```python
n = _global["agent_size"]   # number of active agents
s = _global["step"]         # current step number (1-based)

# Shorthand attribute syntax (rewritten to dict access automatically)
n = _global.agent_size
```

---

## `run_agent()` — call the LLM

```python
from src.run_agent import run_agent   # auto-imported in generated files

_parsed, _raw, _thinking, _tool_calls, _tokens = await run_agent(
    user_input    = "What is 2 + 2?",
    output_config = {"answer": int, "explanation": str},
    agent_state   = _local,
    max_retries   = 3,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `user_input` | `str` | The user message sent to the model |
| `output_config` | `dict[str, type] \| None` | Expected output schema. `None` returns raw string |
| `agent_state` | `dict` | Pass `_local` directly |
| `max_retries` | `int` | JSON enforcement retries (default `3`) |

**Return values:**

| Variable | Type | Description |
|----------|------|-------------|
| `_parsed` | `dict \| str` | Parsed output matching `output_config`, or raw string if `None` |
| `_raw` | `str` | Raw model response text |
| `_thinking` | `list` | Thinking blocks (if model supports it) |
| `_tool_calls` | `list` | Tool call records |
| `_tokens` | `dict` | Token usage (`prompt`, `completion`, `total`) |

The chat history is automatically appended to `_local["agent_config"]["context"]` after each call.

**Structured output examples:**

```python
# Single string
_parsed, *_ = await run_agent("Summarise this.", {"summary": str}, _local)
summary = _parsed["summary"]

# List
_parsed, *_ = await run_agent("List 5 ideas.", {"ideas": list[str]}, _local)

# Raw string (no JSON enforcement)
_parsed, *_ = await run_agent("Write a poem.", None, _local)
poem = _parsed   # plain str
```

---

## `copy_to_workspace()` — stage a file for the agent

Copies any project file into the agent's private workspace so it can be read via the filesystem tool.

```python
from src.operator import copy_to_workspace   # auto-imported in generated files

copy_to_workspace("runs/my_run/data.md", _local)
# → file is now at <workspace_dir>/data.md

copy_to_workspace("runs/my_run/data.md", _local, dest="input/doc.md")
# → file is now at <workspace_dir>/input/doc.md
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `str \| Path` | Source path (absolute or relative to project root) |
| `agent_state` | `dict` | Pass `_local` directly |
| `dest` | `str \| None` | Destination path inside workspace. Defaults to source basename |

Returns the absolute `Path` of the copied file.

---

## `clear_history()` — reset the chat context

Clears the agent's accumulated chat history so the next `run_agent` call starts with a clean conversation.

```python
clear_history(_local)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `agent_state` | `dict` | Pass `_local` directly |

Equivalent to `_local["agent_config"]["context"] = []` but more readable in operator bodies.

---

## Operator types and return values

The operator type is set via the UI dropdown and controls what the engine does after all agents finish the step.

| Type | Base class | Return value | Engine effect |
|------|-----------|--------------|---------------|
| `base` | `Operator` | `None` | State updated in-place |
| `fork` | `ForkOperator` | `int N` | Agent is deep-copied N times |
| `kill` | `KillOperator` | `bool` | `True` removes the agent |
| `sort` | `SortOperator` | `float` | Agents reordered by descending score |
| `shuffle` | `ShuffleOperator` | `(obj, [ranks])` | Cross-agent data exchange |

**Fork example** — split each agent into N children:
```python
return 3   # produce 3 copies of this agent
```

After forking, each child receives:
- `_local["parent_id"]` — `unique_id` of the parent
- `_local["fork_rank"]` — 0-based index among siblings

**Kill example** — keep only high-scoring agents:
```python
return _local["score"] < 0.5   # True = remove this agent
```

**Sort example** — rank agents by score:
```python
return float(_local["score"])  # highest score → rank 0
```

**Shuffle example** — collect outputs from all agents:
```python
my_result = _local["result"]
return (my_result, list(range(_global.agent_size)))
# Each agent receives shuffle_output: dict[rank -> result]
```

---

## Tips

- Variables named with a leading `_` (e.g. `_parsed`, `_tmp`) are never auto-packed into `_local`. Use this for temporaries you don't want to persist.
- `import` statements inside the body are detected and excluded from auto-unpack, so `import json` at the top of your body is safe.
- To persist a variable only sometimes, write to `_local` explicitly: `_local["result"] = x`.
- The agent's chat history accumulates across all `run_agent` calls within a run. Use `clear_history(_local)` to reset it.
