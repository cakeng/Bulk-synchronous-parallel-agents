# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running a step

```bash
# First run — creates engine_state.pt with 1 default agent
python step_engine.py operators/my_op.py

# Subsequent runs — loads existing state, runs operator, overwrites state
python step_engine.py operators/my_op.py

# Custom state file
python step_engine.py operators/my_op.py --state /tmp/my_run.pt
```

## LLM server

vLLM serves a Qwen model locally with an OpenAI-compatible API. Launch it with:

```bash
python launch_vllm.py --config vllm_config.yaml
```

Default endpoint: `http://127.0.0.1:18000/v1`
Default model: set in `vllm_config.yaml` (`model.name`)

## Framework architecture

```
step_engine.py          # CLI entry point — one invocation = one notebook cell
src/
  agent.py              # Agent: isolated variable dict + async LLM interface
  engine.py             # Engine: manages agent list, parallelises operators
  operator.py           # Operator: abstract base class for all steps
  worker.py             # Subprocess worker — one spawned per agent per step
operators/              # User-defined operator files (one Operator subclass each)
engine_state.pt         # Torch-serialised engine state (created at runtime)
```

### Key concepts

**Agent** (`src/agent.py`)
Each agent owns a flat `_vars` dict. Built-in keys always present: `agent_rank`, `llm_state`. `llm_state` contains `base_url`, `api_key`, `model`, and `context` (the chat message list). Access is via `agent["key"]` (raises `KeyError` on missing) or `agent.key` (raises `AttributeError`). The `agent.chat(messages, **kwargs)` async method calls the vLLM server via `openai.AsyncOpenAI`.

**Operator** (`src/operator.py`)
A single step in the execution chain. Define one subclass of `Operator` per file with an `async def run(self, agent)` method. Inside `run`, read agent state with `agent["var"]` and write new state with `agent["var"] = value` — writes persist to `engine_state.pt` after the step.

**Engine** (`src/engine.py`)
Holds the agent list. For each step it serialises every agent to a temp pickle, spawns one `python -m src.worker` subprocess per agent (all concurrent via `asyncio.gather`), then deserialises the updated states back.

**State file** (`engine_state.pt`)
Created/overwritten by `engine.save_state()` via `torch.save`. Contains `{"agents": [<state_dict>, ...]}`. Delete it to reset to a fresh engine.

### Writing an operator

```python
# operators/my_step.py
from src.operator import Operator
from src.run_agent import run_agent

class MyStep(Operator):
    async def run(self, _local, _global):
        parsed, raw, thinking, tool_calls, tokens = await run_agent(
            user_input    = "Hello!",
            output_config = {"reply": str},
            agent_config  = _local["llm_state"],
        )
        _local["last_reply"] = parsed["reply"]    # new variable, persists
```

Exactly one `Operator` subclass per file. The worker finds it by scanning `dir(module)`.
