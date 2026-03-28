"""Engine: manages the list of agents and orchestrates operator execution."""
from __future__ import annotations

import asyncio
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import List

import torch

from .agent import Agent
from . import log

# Default path for the persistent engine state
DEFAULT_STATE_FILE = "engine_state.pt"


class Engine:
    """Manages a list of Agent objects and runs operators across them.

    Lifecycle
    ---------
    1.  ``load_state(path)``   — restore from a previous step's .pt file.
    2.  ``run_operator(path)`` — execute operator on every agent in parallel.
    3.  ``save_state(path)``   — persist updated state for the next step.

    If no state file exists, call ``initialize()`` to create a single default
    agent and begin a fresh execution chain.
    """

    def __init__(self) -> None:
        self.agents: List[Agent] = []

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def initialize(self, llm_config: dict | None = None) -> None:
        """Create a fresh engine with a single default agent (id=0)."""
        self.agents = [Agent(agent_id=0, llm_config=llm_config)]
        log.print_engine("[Engine] Initialized new engine with 1 agent.")

    def load_state(self, path: str = DEFAULT_STATE_FILE) -> None:
        """Restore engine state from a .pt file saved by a previous step."""
        data = torch.load(path, weights_only=False)
        self.agents = []
        for agent_state in data["agents"]:
            agent = Agent(agent_id=agent_state["agent_id"])
            agent.set_state(agent_state)
            self.agents.append(agent)
        log.print_engine(f"[Engine] Loaded {len(self.agents)} agent(s) from '{path}'.")

    def save_state(self, path: str = DEFAULT_STATE_FILE) -> None:
        """Persist current engine state to a .pt file."""
        data = {"agents": [a.get_state() for a in self.agents]}
        torch.save(data, path)
        log.print_engine(f"[Engine] State saved to '{path}'.")

    # ------------------------------------------------------------------
    # Operator execution
    # ------------------------------------------------------------------

    async def run_operator(
        self,
        operator_path: str,
        *,
        verbose: int = 0,
        debug: bool = False,
    ) -> None:
        """Execute the operator on all agents.

        Args:
            operator_path: Path to the operator .py file.
            verbose: 0 = silent, 1 = state summaries, 2 = full variable values.
            debug:   If True, run agents serially instead of concurrently.
        """
        operator_path = str(Path(operator_path).resolve())
        mode = log.debug("SERIAL/DEBUG") if debug else log.dim("parallel")
        log.print_engine(
            f"[Engine] Running '{Path(operator_path).name}' "
            f"on {len(self.agents)} agent(s)  [{mode}]"
        )

        if debug:
            results = []
            for agent in self.agents:
                result = await self._run_agent_subprocess(
                    agent, operator_path, verbose=verbose
                )
                results.append(result)
        else:
            tasks = [
                asyncio.create_task(
                    self._run_agent_subprocess(agent, operator_path, verbose=verbose)
                )
                for agent in self.agents
            ]
            results = await asyncio.gather(*tasks)

        for agent, new_state in zip(self.agents, results):
            agent.set_state(new_state)

        log.print_engine("[Engine] All agents completed.")

    async def _run_agent_subprocess(
        self,
        agent: Agent,
        operator_path: str,
        *,
        verbose: int = 0,
    ) -> dict:
        """Serialize agent state, fork a worker subprocess, return updated state."""
        agent_id = agent["agent_id"]
        prefix = f"[Agent {agent_id}]"
        full = verbose >= 2

        before_state = agent.get_state()

        if verbose:
            log.print_agent_input(agent_id, before_state, full=full)

        # Write agent state to a temp file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fh:
            input_path = fh.name
            pickle.dump(before_state, fh)

        output_path = input_path + ".out.pkl"

        # Run as a module so the project root (not src/) is on sys.path,
        # which avoids shadowing Python's built-in `operator` module.
        project_root = str(Path(__file__).resolve().parent.parent)

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "src.worker",
                "--agent-state", input_path,
                "--operator", operator_path,
                "--output", output_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=project_root,
            )
            stdout, stderr = await proc.communicate()

            if stdout:
                for line in stdout.decode().splitlines():
                    log.print_agent_out(prefix, line)
            if stderr:
                for line in stderr.decode().splitlines():
                    log.print_agent_err(prefix, line)

            if proc.returncode != 0:
                raise RuntimeError(
                    log.error(
                        f"Worker for agent {agent_id} exited with code "
                        f"{proc.returncode}. Check STDERR above."
                    )
                )

            with open(output_path, "rb") as fh:
                new_state = pickle.load(fh)

        finally:
            for p in (input_path, output_path):
                try:
                    os.unlink(p)
                except FileNotFoundError:
                    pass

        if verbose:
            log.print_agent_output_diff(agent_id, before_state, new_state, full=full)

        return new_state
