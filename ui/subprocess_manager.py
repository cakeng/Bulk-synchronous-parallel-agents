"""Subprocess lifecycle management for step_engine.py executions."""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import psutil

from . import state_manager
from .routes.websocket import broadcast

RUNS_DIR = Path("runs")

# Per-run asyncio queues and drain tasks
_item_queues: dict[str, asyncio.Queue] = {}
_drain_tasks: dict[str, asyncio.Task] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def enqueue(run_name: str, operator_name: str) -> None:
    if run_name not in _item_queues:
        _item_queues[run_name] = asyncio.Queue()

    await _item_queues[run_name].put(operator_name)

    run = state_manager.get_run(run_name)
    run.queue.append(operator_name)
    await _emit_queue(run_name)

    task = _drain_tasks.get(run_name)
    if task is None or task.done():
        _drain_tasks[run_name] = asyncio.create_task(_drain_loop(run_name))


async def remove_from_queue(run_name: str, operator_name: str) -> bool:
    """Remove one occurrence of operator_name from the pending queue."""
    run = state_manager.get_run(run_name)
    if operator_name in run.queue:
        run.queue.remove(operator_name)
        # Drain the asyncio.Queue and re-fill without removed item
        q = _item_queues.get(run_name)
        if q:
            items: list[str] = []
            while not q.empty():
                try:
                    items.append(q.get_nowait())
                except asyncio.QueueEmpty:
                    break
            found = False
            for item in items:
                if item == operator_name and not found:
                    found = True
                    continue
                await q.put(item)
        await _emit_queue(run_name)
        return True
    return False


async def kill_current(run_name: str) -> None:
    """Kill the running subprocess (and all its children) for a run."""
    run = state_manager.get_run(run_name)
    pid = run.running_pid
    op_name = run.running_operator
    run.running_pid = None
    run.running_operator = None

    if pid:
        try:
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            parent.kill()
        except psutil.NoSuchProcess:
            pass

    if op_name:
        await broadcast(run_name, {
            "type": "step_failed",
            "run": run_name,
            "operator_name": op_name,
            "error": "Killed by user",
        })


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

async def _emit_queue(run_name: str) -> None:
    run = state_manager.get_run(run_name)
    await broadcast(run_name, {
        "type": "queue_updated",
        "run": run_name,
        "queue": list(run.queue),
    })


async def _drain_loop(run_name: str) -> None:
    q = _item_queues[run_name]
    while not q.empty():
        operator_name = await q.get()
        run = state_manager.get_run(run_name)
        if operator_name in run.queue:
            run.queue.remove(operator_name)
        await _emit_queue(run_name)
        await _run_one(run_name, operator_name)


async def _run_one(run_name: str, operator_name: str) -> None:
    run = state_manager.get_run(run_name)
    run.running_operator = operator_name

    op_path = RUNS_DIR / run_name / "operators" / operator_name
    operator_code = op_path.read_text() if op_path.exists() else ""

    step_num = len(run.engine_state_list) + 1
    timestamp = time.time()

    # Capture pre-step agents from previous record
    pre_agents: list[dict] = []
    if run.engine_state_list:
        pre_agents = list(run.engine_state_list[-1].post_agents)

    await broadcast(run_name, {
        "type": "step_started",
        "run": run_name,
        "operator_name": operator_name,
        "step_num": step_num,
        "timestamp": timestamp,
        "pre_agents": pre_agents,
    })

    project_root = str(Path(__file__).resolve().parent.parent)
    cmd = [sys.executable, "step_engine.py", run_name, operator_name, "--ui-output"]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=project_root,
        )
        run.running_pid = proc.pid

        post_agents: list[dict] = []
        engine_globals: dict = {}

        async def read_stdout() -> None:
            nonlocal post_agents, engine_globals
            assert proc.stdout
            async for raw in proc.stdout:
                line = raw.decode().rstrip()
                try:
                    event = json.loads(line)
                    event["run"] = run_name
                    if event.get("type") == "post_processing_done":
                        post_agents = event.pop("agents", [])
                        engine_globals = event.pop("globals", {})
                    await broadcast(run_name, event)
                except json.JSONDecodeError:
                    await broadcast(run_name, {"type": "log_line", "run": run_name, "text": line})

        async def read_stderr() -> None:
            assert proc.stderr
            async for raw in proc.stderr:
                line = raw.decode().rstrip()
                if line:
                    await broadcast(run_name, {"type": "log_line", "run": run_name, "text": f"[ERR] {line}"})

        await asyncio.gather(read_stdout(), read_stderr())
        await proc.wait()

        run.running_pid = None
        run.running_operator = None

        if proc.returncode == 0 and post_agents:
            record = state_manager.add_engine_state_record(
                run_name=run_name,
                step_num=step_num,
                operator_name=operator_name,
                operator_code=operator_code,
                timestamp=timestamp,
                pre_agents=pre_agents,
                post_agents=post_agents,
                engine_globals=engine_globals,
            )
            await broadcast(run_name, {
                "type": "step_completed",
                "run": run_name,
                "operator_name": operator_name,
                "step_num": step_num,
                "engine_state_uid": record.uid,
                "record": state_manager.record_to_summary(record),
            })
        elif proc.returncode != 0:
            await broadcast(run_name, {
                "type": "step_failed",
                "run": run_name,
                "operator_name": operator_name,
                "error": f"Process exited with code {proc.returncode}",
            })

    except Exception as exc:
        run.running_pid = None
        run.running_operator = None
        await broadcast(run_name, {
            "type": "step_failed",
            "run": run_name,
            "operator_name": operator_name,
            "error": str(exc),
        })
