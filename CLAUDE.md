# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running a step

```bash
# Run against a named run (creates it if it doesn't exist)
python step_engine.py <run_name> <operator>

# Examples
python step_engine.py my_experiment step1.py
python step_engine.py my_experiment operators/step1.py  # explicit path

# Flags
python step_engine.py my_experiment step1.py --debug   # serial execution
python step_engine.py my_experiment step1.py --verbose 2
```

## LLM server

vLLM serves a Qwen model locally with an OpenAI-compatible API. Launch it with:

```bash
python launch_vllm.py --config vllm_config.yaml
```

Default endpoint: `http://127.0.0.1:18000/v1`
Default model: set in `vllm_config.yaml` (`model.name`)

## UI server

```bash
python -m ui.server   # starts at http://127.0.0.1:18001
```

The UI provides a visual execution tree, operator editor, agent state inspector, and chat context viewer.

## Framework architecture

```
step_engine.py          # CLI entry point — one invocation = one notebook cell
src/
  agent.py              # Agent: isolated variable dict + async LLM interface
  engine.py             # Engine: manages agent list, parallelises operators
  operator.py           # Operator: abstract base class for all steps
  worker.py             # Subprocess worker — one spawned per agent per step
  run_agent.py          # run_agent(): structured LLM call with retries & call_log
  context.py            # Context: thin wrapper around chat message list
  tools.py              # Dynamic tool registry (scans tools/*.py)
  op_codegen.py         # Operator code generation and body extraction
operators/              # User-defined operator files (one Operator subclass each)
tools/                  # Tool modules (filesystem, web, bash, git, code, hf_paper)
ui/                     # FastAPI UI server
runs/
  <run_name>/
    engine_states/      # Per-step .pt state snapshots
    operators/          # Operator files for this run
    workspaces/
      <unique_id>/      # Per-agent workspace directory (files, tool cache)
```

### Key concepts

**Agent** (`src/agent.py`)
Each agent owns a flat `_vars` dict. Built-in keys always present: `agent_rank`, `agent_config`. `agent_config` contains `base_url`, `api_key`, `model`, `context` (the chat message list), and `call_log` (per-call thinking/tool/token history). Access is via `agent["key"]` (raises `KeyError` on missing) or `agent.key` (raises `AttributeError`). The `agent.chat(messages, **kwargs)` async method calls the vLLM server via `openai.AsyncOpenAI`.

**Operator** (`src/operator.py`)
A single step in the execution chain. Define one subclass of `Operator` per file with an `async def run(self, _local, _global)` method. Inside `run`, read and write agent state directly via `_local["var"]`. Writes persist to the run's engine state after the step. `_global` contains engine-wide variables (`step`, `agent_size`, …).

**Engine** (`src/engine.py`)
Holds the agent list. For each step it serialises every agent to a temp pickle, spawns one `python -m src.worker` subprocess per agent (all concurrent via `asyncio.gather`), then deserialises the updated states back. Special operator types (fork/kill/sort/shuffle) restructure the agent list after all subprocesses complete.

**Run directory** (`runs/<run_name>/`)
Created automatically on first use. Contains `engine_states/` (snapshot `.pt` files), `operators/` (operator `.py` files), and `workspaces/<unique_id>/` (per-agent file sandbox). The `unique_id` is derived from agent state so it remains stable across rank reassignments.

### Writing an operator

```python
# runs/my_experiment/operators/my_step.py
from src.operator import Operator
from src.run_agent import run_agent

class MyStep(Operator):
    async def run(self, _local, _global):
        parsed, raw, thinking, tool_calls, tokens = await run_agent(
            user_input    = "Hello!",
            output_config = {"reply": str},
            agent_state   = _local,          # pass the full _local dict
        )
        _local["last_reply"] = parsed["reply"]    # new variable, persists
```

Exactly one `Operator` subclass per file. The worker finds it by scanning `dir(module)`.

### run_agent()

```python
from src.run_agent import run_agent

parsed, raw, thinking, tool_calls, tokens = await run_agent(
    user_input    = "What is 2 + 2?",
    output_config = {"answer": int, "explanation": str},  # None = raw string
    agent_state   = _local,   # full _local dict; must contain agent_config
    max_retries   = 3,        # JSON enforcement retries (structured output only)
)
```

- `agent_state["agent_config"]` must contain `base_url`, `api_key`, `model`.
- On each successful call, `agent_config["context"]` is updated with the new messages and `agent_config["call_log"]` gets a new entry: `{"thinking": [...], "tool_calls": [...], "tokens": {...}}`.
- `output_config=None` skips JSON enforcement and returns the raw string.
- `output_config` supports generic types: `list[str]`, `dict[str, int]`, etc.

### Tools

Tools are loaded automatically from `tools/*.py` when an agent has a `workspace_dir`. Each tool module exposes `TOOL_NAME`, `TOOL_PROMPTS["full"]`, `TOOL_SCHEMAS["full"]`, and `dispatch(payload, workspace_dir)`.

Available tools:
- **filesystem** — read, write, list, delete files within the agent workspace
- **bash** — run shell commands in the workspace directory
- **web** — DuckDuckGo search, fetch URLs as Markdown (cached)
- **git** — init, clone, status, commit, diff, log within workspace repos
- **code** — run Python/shell code, manage a per-workspace `.venv`
- **hf_paper** — list, search, get info, and read HuggingFace Daily Papers

Dispatch from an operator:
```python
from src import tools
result = tools.dispatch("filesystem", {"action": "read_file", "path": "out.txt"}, workspace_dir)
result = tools.dispatch_function_call("hf_paper__search", {"query": "diffusion models"}, workspace_dir)
```

### Operator types

| Type    | Base class      | `run()` return  | Engine effect                        |
|---------|----------------|-----------------|--------------------------------------|
| base    | Operator        | None            | State updated in-place               |
| fork    | ForkOperator    | int (N)         | Each agent spawns N deep-copied children |
| kill    | KillOperator    | bool            | True → agent removed (all-kill refused) |
| sort    | SortOperator    | float (score)   | Agents reordered by descending score |
| shuffle | ShuffleOperator | (obj, [ranks])  | Cross-agent data distribution        |

### UI

The browser UI at `http://127.0.0.1:18001` provides:
- **Execution Tree** — visual tree of all steps; fork/kill/sort/shuffle edges colour-coded; in-progress and queued steps shown live; state restored on browser refresh
- **Operator Editor** — Monaco-based editor; run individual operators or a full queue; drag-to-reorder
- **Engine State** panel — engine globals and operator source for the selected step
- **Agent State** panel — per-agent variable table for the selected step/agent
- **Chat Context** panel — rendered chat messages with **Show Thinking** and **Show Tool Calls** toggles; thinking shown in purple, tool calls in orange, tool results in teal
