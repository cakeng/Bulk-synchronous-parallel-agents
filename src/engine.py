"""Engine: manages the list of agents and orchestrates operator execution."""
from __future__ import annotations

import asyncio
import copy
import heapq
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import torch

from .agent import Agent, compute_unique_id
from . import log, tools as _tools

DEFAULT_STATE_FILE = "engine_state.pt"

# Keys injected by the engine into each agent's vars before operator execution.
# Stripped back out after the subprocess returns.
_ENGINE_KEYS = frozenset({"engine_globals"})


class Engine:
    """Manages a list of Agent objects and runs operators across them.

    engine_globals always contains the reserved key ``step`` (int), incremented
    at the start of every ``run_operator`` call.  All other keys are user-defined.

    Special operator types (ForkOperator, KillOperator, SortOperator,
    ShuffleOperator) trigger engine-level restructuring of the agent list after
    all subprocesses have been joined.  Agent IDs are reassigned sequentially
    after any structural change.
    """

    def __init__(self) -> None:
        self.agents: List[Agent] = []
        self.globals: Dict[str, Any] = {"step": 0}
        self.workspace_base: Path | None = None  # set by step_engine before initialize()
        # Tool-slot semaphore state — re-initialised at the start of every run_operator call
        self._tool_slots_available: int = 10
        self._tool_waiters: list = []   # min-heap of (rank, uid, asyncio.Event)
        self._tool_waiter_uid: int = 0  # monotonic counter used as heap tiebreaker

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def initialize(self, llm_state: dict | None = None) -> None:
        self.agents = [Agent(agent_rank=0, llm_state=llm_state)]
        self.globals = {"step": 0, "agent_size": 1}
        self._setup_agent_workspace(self.agents[0])
        log.print_engine("[Engine] Initialized new engine with 1 agent.")

    def load_state(self, path: str = DEFAULT_STATE_FILE) -> None:
        data = torch.load(path, weights_only=False)
        self.agents = []
        for agent_state in data["agents"]:
            agent = Agent(agent_rank=agent_state["agent_rank"])
            agent.set_state(agent_state)
            self.agents.append(agent)
        self.globals = data.get("globals", {"step": 0, "agent_size": len(self.agents)})
        log.print_engine(
            f"[Engine] Loaded {len(self.agents)} agent(s) from '{path}'  "
            f"[step={self.globals.get('step', 0)}]."
        )

    def save_state(self, path: str = DEFAULT_STATE_FILE) -> None:
        data = {
            "agents":  [a.get_state() for a in self.agents],
            "globals": self.globals,
        }
        torch.save(data, path)
        log.print_engine(f"[Engine] State saved to '{path}'.")

    # ------------------------------------------------------------------
    # Operator execution
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Tool-slot priority semaphore
    # ------------------------------------------------------------------

    async def _acquire_tool_slot(self, rank: int, stdin) -> None:
        """Block until a tool-execution slot is available, granting priority to lower ranks."""
        if self._tool_slots_available > 0:
            self._tool_slots_available -= 1
            stdin.write(b"go\n")
            await stdin.drain()
            return
        event = asyncio.Event()
        uid = self._tool_waiter_uid
        self._tool_waiter_uid += 1
        heapq.heappush(self._tool_waiters, (rank, uid, event))
        try:
            await event.wait()
        except asyncio.CancelledError:
            # Subprocess was killed while waiting — remove from the heap and propagate.
            self._tool_waiters = [t for t in self._tool_waiters if t[2] is not event]
            heapq.heapify(self._tool_waiters)
            raise
        stdin.write(b"go\n")
        await stdin.drain()

    async def _release_tool_slot(self) -> None:
        """Return one slot; wake the lowest-rank waiting agent if any."""
        if self._tool_waiters:
            _, _, event = heapq.heappop(self._tool_waiters)
            event.set()
        else:
            self._tool_slots_available += 1

    # ------------------------------------------------------------------
    # Operator execution
    # ------------------------------------------------------------------

    async def run_operator(
        self,
        operator_path: str,
        *,
        verbose: int = 0,
        debug: bool = False,
        ui_callback=None,   # async callable(event: dict) | None
    ) -> None:
        self.globals["step"] = self.globals.get("step", 0) + 1
        self.globals["agent_size"] = len(self.agents)
        # Re-initialise tool-slot semaphore for this step
        limit = max(1, int(self.globals.get("concurrent_tool_call_limit", 10)))
        self._tool_slots_available = limit
        self._tool_waiters = []
        self._tool_waiter_uid = 0
        operator_path = str(Path(operator_path).resolve())

        engine_vars: Dict[str, Any] = {"engine_globals": dict(self.globals)}

        mode = log.debug("SERIAL/DEBUG") if debug else log.dim("parallel")
        log.print_engine(
            f"[Engine] Step {self.globals['step']} — running "
            f"'{Path(operator_path).name}' on {len(self.agents)} agent(s)  [{mode}]"
        )

        # Notify UI that agents are starting
        if ui_callback:
            for agent in self.agents:
                await ui_callback({"type": "agent_started", "agent_rank": agent["agent_rank"]})

        if debug:
            results = []
            for agent in self.agents:
                result = await self._run_agent_subprocess(
                    agent, operator_path, engine_vars=engine_vars,
                    verbose=verbose, ui_callback=ui_callback,
                )
                results.append(result)
        else:
            tasks = [
                asyncio.create_task(
                    self._run_agent_subprocess(
                        agent, operator_path, engine_vars=engine_vars,
                        verbose=verbose, ui_callback=ui_callback,
                    )
                )
                for agent in self.agents
            ]
            results = await asyncio.gather(*tasks)

        # All worker outputs: {"state": ..., "return_value": ..., "operator_type": ...}
        op_type = results[0]["operator_type"] if results else "base"

        # Merge globals and apply new states to agents
        merged_globals = dict(self.globals)
        new_states: List[dict] = []
        return_values: List[Any] = []
        for result in results:
            state = result["state"]
            merged_globals.update(state.pop("engine_globals", {}))
            new_states.append(state)
            return_values.append(result["return_value"])

        self.globals = merged_globals
        for agent, state in zip(self.agents, new_states):
            agent.set_state(state)

        # Operator-type post-processing (runs after all subprocesses joined)
        if op_type == "fork":
            self._apply_fork(return_values)
        elif op_type == "kill":
            self._apply_kill(return_values)
        elif op_type == "sort":
            self._apply_sort(return_values)
        elif op_type == "shuffle":
            self._apply_shuffle(return_values)

        self.globals["agent_size"] = len(self.agents)
        log.print_engine(
            f"[Engine] All agents completed. "
            f"({len(self.agents)} agent(s) active)"
        )

        # Notify UI with final agent list after all post-processing
        if ui_callback:
            await ui_callback({
                "type": "post_processing_done",
                "agents": [a.get_state() for a in self.agents],
                "globals": dict(self.globals),
            })

    # ------------------------------------------------------------------
    # Subprocess dispatch
    # ------------------------------------------------------------------

    async def _run_agent_subprocess(
        self,
        agent: Agent,
        operator_path: str,
        *,
        engine_vars: Dict[str, Any],
        verbose: int = 0,
        ui_callback=None,
    ) -> dict:
        """Returns the raw worker output dict:
        {"state": ..., "return_value": ..., "operator_type": ...}
        """
        agent_rank = agent["agent_rank"]
        prefix     = f"[Agent {agent_rank}]"
        full       = verbose >= 2

        before_state = agent.get_state()

        send_state = {**before_state, **engine_vars}

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fh:
            input_path = fh.name
            pickle.dump(send_state, fh)

        output_path  = input_path + ".out.pkl"
        project_root = str(Path(__file__).resolve().parent.parent)

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "src.worker",
                "--agent-state", input_path,
                "--operator",    operator_path,
                "--output",      output_path,
                stdin=asyncio.subprocess.PIPE,   # used for tool-slot grant signals
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=project_root,
                limit=8 * 1024 * 1024,  # 8 MB — avoids "chunk longer than limit" on large tool outputs
            )

            stderr_lines: list[str] = []

            async def _stream_stdout() -> None:
                assert proc.stdout
                async for raw in proc.stdout:
                    line = raw.decode().rstrip()
                    log.print_agent_out(prefix, line)
                    if ui_callback:
                        try:
                            event = json.loads(line)
                            if isinstance(event, dict) and "type" in event:
                                etype = event.get("type")
                                if etype == "tool_slot_request":
                                    await self._acquire_tool_slot(agent_rank, proc.stdin)
                                    continue
                                if etype == "tool_slot_release":
                                    await self._release_tool_slot()
                                    continue
                                event["agent_rank"] = agent_rank
                                await ui_callback(event)
                                continue
                        except (json.JSONDecodeError, TypeError):
                            pass
                        await ui_callback({"type": "agent_log", "agent_rank": agent_rank, "stream": "stdout", "text": line})

            async def _stream_stderr() -> None:
                assert proc.stderr
                async for raw in proc.stderr:
                    line = raw.decode().rstrip()
                    stderr_lines.append(line)
                    log.print_agent_err(prefix, line)
                    if ui_callback:
                        await ui_callback({"type": "agent_log", "agent_rank": agent_rank, "stream": "stderr", "text": line})

            # Run stream tasks concurrently; wait for the process to exit first,
            # then drain streams (they finish naturally on EOF).  Force-cancel after
            # 2 s to unblock any task stuck in _acquire_tool_slot when the
            # subprocess was killed before a slot was granted.
            stdout_task = asyncio.create_task(_stream_stdout())
            stderr_task = asyncio.create_task(_stream_stderr())
            await proc.wait()
            try:
                await asyncio.wait_for(
                    asyncio.gather(stdout_task, stderr_task, return_exceptions=True),
                    timeout=2.0,
                )
            except asyncio.TimeoutError:
                stdout_task.cancel()
                stderr_task.cancel()
                await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

            if proc.returncode != 0:
                error_text = "\n".join(stderr_lines).strip()
                if not error_text:
                    error_text = f"Worker for agent {agent_rank} exited with code {proc.returncode}"
                if ui_callback:
                    await ui_callback({
                        "type": "agent_failed",
                        "agent_rank": agent_rank,
                        "error": error_text,
                    })
                raise RuntimeError(
                    log.error(
                        f"Worker for agent {agent_rank} exited with code "
                        f"{proc.returncode}. Check STDERR above."
                    )
                )

            with open(output_path, "rb") as fh:
                worker_output: dict = pickle.load(fh)

            # Notify UI that this agent finished
            if ui_callback:
                slim_state = {
                    k: v for k, v in worker_output["state"].items()
                    if k not in _ENGINE_KEYS
                }
                await ui_callback({
                    "type": "agent_completed",
                    "agent_rank": agent_rank,
                    "state": slim_state,
                })

        finally:
            for p in (input_path, output_path):
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass

        if verbose:
            new_state    = worker_output["state"]
            agent_after  = {k: v for k, v in new_state.items() if k not in _ENGINE_KEYS}
            log.print_agent_output_diff(agent_rank, before_state, agent_after, full=full)

            globals_before = engine_vars["engine_globals"]
            globals_after  = new_state.get("engine_globals", globals_before)
            if globals_after != globals_before:
                log.print_globals_diff(globals_before, globals_after, full=full)

        return worker_output

    # ------------------------------------------------------------------
    # Post-processing for special operator types
    # ------------------------------------------------------------------

    def _setup_agent_workspace(self, agent: Agent) -> None:
        """Create the workspace directory for an agent and populate agent_config with tools."""
        if not self.workspace_base:
            return
        uid = agent["unique_id"]
        ws = self.workspace_base / uid
        ws.mkdir(parents=True, exist_ok=True)
        agent["workspace_dir"] = str(ws)
        agent["agent_config"]["tools"] = _tools.build_tool_schemas()

    def _reindex_agents(self) -> None:
        """Reassign sequential agent_ranks based on current list order."""
        for new_rank, agent in enumerate(self.agents):
            agent["agent_rank"] = new_rank

    def _apply_fork(self, return_values: List[int]) -> None:
        """Expand the agent list: each agent produces N deep-copied children."""
        new_agents: List[Agent] = []
        for agent, n in zip(self.agents, return_values):
            parent_uid = agent["unique_id"]
            for fork_rank in range(n):
                state = copy.deepcopy(agent.get_state())
                state["parent_id"]  = parent_uid
                state["fork_rank"]  = fork_rank
                state["unique_id"] = compute_unique_id(state)
                child = Agent(agent_rank=0)   # temp rank; reindexed below
                child.set_state(state)
                self._setup_agent_workspace(child)  # new workspace per child uid
                new_agents.append(child)
        self.agents = new_agents
        self._reindex_agents()
        log.print_engine(
            f"[Engine] Fork complete: {len(self.agents)} agent(s) produced."
        )

    def _apply_kill(self, return_values: List[bool]) -> None:
        """Remove agents that returned True; refuse if all would be killed."""
        if all(return_values):
            raise RuntimeError(
                "KillOperator: all agents returned True — "
                "refusing to eliminate the entire agent list."
            )
        n_killed = sum(return_values)
        self.agents = [a for a, kill in zip(self.agents, return_values) if not kill]
        self._reindex_agents()
        log.print_engine(
            f"[Engine] Kill complete: {n_killed} removed, "
            f"{len(self.agents)} remaining."
        )

    def _apply_sort(self, return_values: List[float]) -> None:
        """Reorder agents by descending score."""
        paired = sorted(
            zip(return_values, self.agents),
            key=lambda x: x[0],
            reverse=True,
        )
        self.agents = [a for _, a in paired]
        self._reindex_agents()
        scores = [rv for rv, _ in paired]
        log.print_engine(f"[Engine] Sort complete: scores {scores}.")

    def _apply_shuffle(self, return_values: List[tuple]) -> None:
        """Distribute shared objects; populate each agent's ``shuffle_output``."""
        # Build lookup: agent_rank -> shared object
        rank_to_obj: Dict[int, Any] = {
            agent["agent_rank"]: rv[0]
            for agent, rv in zip(self.agents, return_values)
        }
        for agent, (_, requested_ranks) in zip(self.agents, return_values):
            shuffle_output: Dict[int, Any] = {}
            for rank in requested_ranks:
                if rank not in rank_to_obj:
                    raise KeyError(
                        f"ShuffleOperator: agent_rank {rank} not found "
                        "among current agents."
                    )
                shuffle_output[rank] = copy.deepcopy(rank_to_obj[rank])
            agent["shuffle_output"] = shuffle_output
        log.print_engine("[Engine] Shuffle complete: outputs distributed.")
