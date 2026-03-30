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
import json
import os
import pickle
import sys

from src.agent import AgentState
from src import ipc
from src.operator import (
    Operator, ForkOperator, KillOperator, SortOperator, ShuffleOperator,
)


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
        and obj not in (ForkOperator, KillOperator, SortOperator, ShuffleOperator)
    ]
    if not candidates:
        raise ValueError(
            f"No concrete Operator subclass found in '{operator_path}'. "
            "Define a class that inherits from one of the Operator types in src.operator."
        )
    if len(candidates) > 1:
        names = [c.__name__ for c in candidates]
        raise ValueError(
            f"Multiple Operator subclasses found in '{operator_path}': {names}. "
            "Define exactly one per operator file."
        )
    return candidates[0]()


def _validate_return(op_type: str, rv) -> None:
    """Raise TypeError if ``rv`` does not match the expected type for ``op_type``."""
    if op_type == "base":
        if rv is not None:
            raise TypeError(
                f"Operator (base) must return None, got {type(rv).__name__!r}: {rv!r}."
            )
    elif op_type == "fork":
        if isinstance(rv, bool) or not isinstance(rv, int) or rv < 0:
            raise TypeError(
                f"ForkOperator must return a non-negative int, "
                f"got {type(rv).__name__!r}: {rv!r}."
            )
    elif op_type == "kill":
        if not isinstance(rv, bool):
            raise TypeError(
                f"KillOperator must return a bool, "
                f"got {type(rv).__name__!r}: {rv!r}."
            )
    elif op_type == "sort":
        if isinstance(rv, bool) or not isinstance(rv, (int, float)):
            raise TypeError(
                f"SortOperator must return a float (or int), "
                f"got {type(rv).__name__!r}: {rv!r}."
            )
    elif op_type == "shuffle":
        if (
            not isinstance(rv, tuple)
            or len(rv) != 2
            or not isinstance(rv[1], list)
        ):
            raise TypeError(
                f"ShuffleOperator must return (Any, list[agent_rank]), "
                f"got {type(rv).__name__!r}: {rv!r}."
            )


async def _main() -> None:
    parser = argparse.ArgumentParser(description="BSA agent worker subprocess")
    parser.add_argument("--agent-state", required=True)
    parser.add_argument("--operator",    required=True)
    parser.add_argument("--output",      required=True)
    args = parser.parse_args()

    with open(args.agent_state, "rb") as fh:
        state: dict = pickle.load(fh)

    # Split the flat state dict into the two operator arguments:
    #   _local  — agent-specific variables (wrapped in AgentState for helpers)
    #   _global — engine-global variables (injected by the engine)
    _global_dict: dict = state.pop("engine_globals", {})
    _local_dict = AgentState(state)
    agent_rank  = int(_local_dict.get("agent_rank", 0))

    # Connect to engine TCP server when spawned by the engine
    _tcp_port = os.environ.get("BSA_ENGINE_TCP_PORT")
    if _tcp_port:
        try:
            _tcp_reader, _tcp_writer = await asyncio.open_connection("127.0.0.1", int(_tcp_port))
            # Identify this worker to the engine
            _tcp_writer.write((json.dumps({"type": "hello", "agent_rank": agent_rank}) + "\n").encode())
            await _tcp_writer.drain()
            ipc.configure(_tcp_reader, _tcp_writer)
        except Exception as _tcp_err:
            print(f"[worker] TCP connect failed: {_tcp_err}", file=sys.stderr)

    try:
        operator  = _load_operator(args.operator)
        op_type   = operator.OPERATOR_TYPE

        rv = await operator.run(_local_dict, _global_dict)
        _validate_return(op_type, rv)

        # Rejoin so the engine can strip and merge engine_globals on return
        result_state = {**_local_dict, "engine_globals": _global_dict}
        output = {
            "state":         result_state,
            "return_value":  rv,
            "operator_type": op_type,
        }
        with open(args.output, "wb") as fh:
            pickle.dump(output, fh)
    finally:
        await ipc.close()


if __name__ == "__main__":
    asyncio.run(_main())
