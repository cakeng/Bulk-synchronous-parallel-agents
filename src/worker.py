#!/usr/bin/env python
"""Subprocess worker: loads one agent's state, runs the operator, saves result.

Called by Engine._run_agent_subprocess().  Not intended to be used directly.

Usage:
    python -m src.worker \\
        --agent-state /tmp/agent_0_in.pkl \\
        --operator operators/my_step.py \\
        --output /tmp/agent_0_out.pkl
"""
import argparse
import asyncio
import importlib.util
import pickle
import sys

from src.agent import Agent
from src.operator import Operator


def _load_operator(operator_path: str) -> Operator:
    """Import operator_path and return an instance of the Operator subclass."""
    spec = importlib.util.spec_from_file_location("_operator_module", operator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load operator from '{operator_path}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    candidates = [
        obj
        for name in dir(module)
        for obj in [getattr(module, name)]
        if isinstance(obj, type)
        and issubclass(obj, Operator)
        and obj is not Operator
    ]
    if not candidates:
        raise ValueError(
            f"No Operator subclass found in '{operator_path}'. "
            "Define a class that inherits from src.operator.Operator."
        )
    if len(candidates) > 1:
        names = [c.__name__ for c in candidates]
        raise ValueError(
            f"Multiple Operator subclasses found in '{operator_path}': {names}. "
            "Define exactly one per operator file."
        )
    return candidates[0]()


async def _main() -> None:
    parser = argparse.ArgumentParser(description="BSA agent worker subprocess")
    parser.add_argument("--agent-state", required=True, help="Pickle file with agent state dict")
    parser.add_argument("--operator", required=True, help="Path to operator .py file")
    parser.add_argument("--output", required=True, help="Pickle file to write updated agent state")
    args = parser.parse_args()

    # Load agent state
    with open(args.agent_state, "rb") as fh:
        state: dict = pickle.load(fh)

    agent = Agent(agent_id=state["agent_id"])
    agent.set_state(state)

    # Load and run operator
    operator = _load_operator(args.operator)
    await operator.run(agent)

    # Write updated state
    with open(args.output, "wb") as fh:
        pickle.dump(agent.get_state(), fh)


if __name__ == "__main__":
    asyncio.run(_main())
