#!/usr/bin/env python
"""step_engine.py — entry point for the BSA agentic framework.

Usage
-----
    # Run an operator against a named run (creates the run if it doesn't exist)
    python step_engine.py <run_name> <operator>

    # <operator> is resolved in order:
    #   1. As-is if it contains a path separator (absolute or CWD-relative)
    #   2. Otherwise looked up in runs/<run_name>/operators/<operator>

    python step_engine.py my_experiment step1.py
    python step_engine.py my_experiment operators/step1.py   # explicit path

    # Flags
    python step_engine.py my_experiment step1.py --debug --verbose 2

Run directory layout
--------------------
    runs/
      <run_name>/
        engine_state.pt   ← persisted agent state
        operators/        ← operator files for this run

Each invocation corresponds to one "cell" in the notebook-style execution model:
  load state → run operator across all agents → save state
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

from src.engine import Engine
from src import log

RUNS_DIR = Path("runs")


def resolve_run(run_name: str) -> Path:
    """Return the run directory, creating it (with an operators/ subdir) if needed."""
    run_dir = RUNS_DIR / run_name
    if not run_dir.exists():
        (run_dir / "operators").mkdir(parents=True)
        log.print_engine(f"[step_engine] Created new run '{run_name}' at '{run_dir}/'.")
    return run_dir


def resolve_operator(op_arg: str, run_dir: Path) -> Path:
    """Resolve the operator file path.

    If op_arg contains a path separator it is used as-is (absolute or relative
    to CWD).  Otherwise it is looked up inside run_dir/operators/.
    """
    if os.sep in op_arg or "/" in op_arg:
        path = Path(op_arg)
    else:
        path = run_dir / "operators" / op_arg
    if not path.is_file():
        log.print_error(f"[step_engine] ERROR: operator file not found: '{path}'")
        sys.exit(1)
    return path.resolve()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="BSA Step Engine — run one operator step across all agents."
    )
    parser.add_argument(
        "run",
        help="Name of the run (matches runs/<run>/). Created automatically if absent.",
    )
    parser.add_argument(
        "operator",
        help=(
            "Operator file to execute. Bare filename → looked up in "
            "runs/<run>/operators/. Path with separators → used as-is."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        type=int, nargs="?", const=1, default=0,
        metavar="LEVEL",
        help="Verbosity level: 1 = show state summaries, 2 = show full variable values.",
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Run agents serially (one at a time) instead of concurrently.",
    )
    args = parser.parse_args()

    if args.debug:
        log.print_debug("[step_engine] DEBUG mode — agents will run serially.")

    run_dir   = resolve_run(args.run)
    op_path   = resolve_operator(args.operator, run_dir)
    state_file = str(run_dir / "engine_state.pt")

    engine = Engine()

    if os.path.isfile(state_file):
        engine.load_state(state_file)
    else:
        log.print_engine(f"[step_engine] No state file found. Initializing fresh engine for run '{args.run}'.")
        engine.initialize()

    if args.verbose:
        _print_engine_overview(engine, run_name=args.run, verbose_level=args.verbose)

    await engine.run_operator(
        str(op_path),
        verbose=args.verbose,
        debug=args.debug,
    )

    if args.verbose:
        _print_engine_overview(engine, run_name=args.run, verbose_level=args.verbose)

    engine.save_state(state_file)


def _print_engine_overview(engine: Engine, run_name: str, verbose_level: int = 1) -> None:
    """Print a color-coded summary of the engine and all agent states."""
    n = len(engine.agents)
    full = verbose_level >= 2
    header = f"run={run_name}  step={engine.globals.get('step', 0)}  {'1 agent' if n == 1 else f'{n} agents'}"
    print(log.engine(f"\n  ╔══ ENGINE STATE ({header}) {'═' * max(0, 38 - len(header))}"))

    # Engine-level globals
    if engine.globals:
        print(log.engine(f"  ║"))
        print(log.engine(f"  ║  {log.bold('engine_globals')}"))
        for k, v in engine.globals.items():
            lines = log.format_value(v, full=full).splitlines()
            print(log.engine(f"  ║    {log.bold(k)}: {log.dim(lines[0])}"))
            for extra in lines[1:]:
                print(log.engine(f"  ║      {log.dim(extra)}"))
    else:
        print(log.engine(f"  ║  {log.dim('engine_globals: (empty)')}"))

    # Per-agent state
    for agent in engine.agents:
        state = agent.get_state()
        rank = state.get("agent_rank", "?")
        cfg = state.get("llm_state", {})
        print(log.engine(f"  ║"))
        print(log.engine(f"  ║  Agent {rank}  —  {cfg.get('base_url', '?')}  model={cfg.get('model', '?')}"))
        for k, v in state.items():
            if k in ("agent_rank", "llm_state"):
                continue
            lines = log.format_value(v, full=full).splitlines()
            print(log.engine(f"  ║    {log.bold(k)}: {log.dim(lines[0])}"))
            for extra in lines[1:]:
                print(log.engine(f"  ║      {log.dim(extra)}"))
    print(log.engine(f"  ╚══════════════════════════════════════════════════\n"))


if __name__ == "__main__":
    asyncio.run(main())
