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

async def enqueue(run_name: str, operator_name: str, kill_failed: bool = False) -> None:
    if run_name not in _item_queues:
        _item_queues[run_name] = asyncio.Queue()

    await _item_queues[run_name].put({"name": operator_name, "kill_failed": kill_failed})

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


async def kill_agent(run_name: str, agent_rank: int) -> bool:
    """Kill the worker subprocess for a specific agent. Returns True if a PID was found."""
    run = state_manager.get_run(run_name)
    pid = run.agent_pids.get(agent_rank)
    if not pid:
        return False
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
    run.agent_pids.pop(agent_rank, None)
    run.killed_ranks.add(agent_rank)
    return True


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

# Event types that should be buffered in step_log for reconnect replay.
# Keep the per-agent lifecycle events (small in number) but exclude
# agent_log (can be thousands of lines) and agent_completed (large state
# payload) to avoid flooding the client on WS reconnect, which would
# starve Monaco's cursor blink animation of animation frames.
_LOG_TYPES = frozenset({
    "step_started",
    "agent_started",
    "agent_failed",
    "step_failed",
})


async def _emit_queue(run_name: str) -> None:
    run = state_manager.get_run(run_name)
    await broadcast(run_name, {
        "type": "queue_updated",
        "run": run_name,
        "queue": list(run.queue),
    })


async def _broadcast_and_log(run_name: str, event: dict) -> None:
    """Broadcast an event and, if it's a step-lifecycle event, buffer it for reconnect."""
    if event.get("type") in _LOG_TYPES:
        run = state_manager.get_run(run_name)
        run.step_log.append(event)
    await broadcast(run_name, event)


async def _drain_loop(run_name: str) -> None:
    q = _item_queues[run_name]
    while not q.empty():
        item = await q.get()
        operator_name = item["name"]
        kill_failed   = item.get("kill_failed", False)
        run = state_manager.get_run(run_name)
        if operator_name in run.queue:
            run.queue.remove(operator_name)
        await _emit_queue(run_name)
        await _run_one(run_name, operator_name, kill_failed=kill_failed)


async def _run_one(run_name: str, operator_name: str, kill_failed: bool = False) -> None:
    run = state_manager.get_run(run_name)
    run.running_operator = operator_name
    run.step_log = []    # clear previous step log
    run.agent_logs = {}  # clear previous agent logs
    run.agent_pids = {}  # clear previous agent PIDs
    run.killed_ranks = set()  # clear killed ranks for this step

    op_path = RUNS_DIR / run_name / "operators" / operator_name
    operator_code = op_path.read_text() if op_path.exists() else ""

    step_num = len(run.engine_state_list) + 1
    timestamp = time.time()

    # Capture pre-step agents from previous record
    pre_agents: list[dict] = []
    if run.engine_state_list:
        pre_agents = list(run.engine_state_list[-1].post_agents)

    await _broadcast_and_log(run_name, {
        "type": "step_started",
        "run": run_name,
        "operator_name": operator_name,
        "step_num": step_num,
        "timestamp": timestamp,
        "pre_agents": pre_agents,
    })

    project_root = str(Path(__file__).resolve().parent.parent)
    cmd = [sys.executable, "step_engine.py", run_name, operator_name, "--ui-output"]
    if kill_failed:
        cmd.append("--kill-failed")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=project_root,
            limit=256 * 1024 * 1024,  # 256 MB — stdout is debug-only in TCP mode
        )
        run.running_pid = proc.pid

        post_agents: list[dict] = []
        engine_killed_agents: list[dict] = []
        engine_globals: dict = {}
        failed_ranks: set = set()
        proc_stderr_lines: list[str] = []

        # Broadcast queue: decouple pipe-reading from WebSocket sends so a slow
        # WebSocket client never backs up the pipe to step_engine.
        _bcast_queue: asyncio.Queue = asyncio.Queue()

        async def _drain_bcast_queue() -> None:
            while True:
                item = await _bcast_queue.get()
                if item is None:  # sentinel: stop draining
                    break
                await _broadcast_and_log(run_name, item)

        bcast_task = asyncio.create_task(_drain_bcast_queue())

        # Scan stdout lines until we find {"type":"tcp_ready","port":N}.
        # step_engine may emit colored log messages to stdout before this line
        # (e.g. "No state file found"), so we skip non-JSON / non-tcp_ready lines.
        _tcp_reader: asyncio.StreamReader | None = None
        _tcp_writer: asyncio.StreamWriter | None = None
        assert proc.stdout
        try:
            deadline = asyncio.get_event_loop().time() + 30.0
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                raw = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
                if not raw:
                    break
                line = raw.decode().rstrip()
                try:
                    data = json.loads(line)
                    if data.get("type") == "tcp_ready":
                        _tcp_reader, _tcp_writer = await asyncio.open_connection(
                            "127.0.0.1", data["port"],
                            limit=8 * 1024 * 1024 * 1024,  # 8 GB — carries all agent states
                        )
                        break
                except json.JSONDecodeError:
                    pass  # skip colored log lines printed before tcp_ready
        except Exception as _tcp_err:
            print(f"[subprocess_manager] TCP connect failed: {_tcp_err}", file=sys.stderr)

        async def read_tcp() -> None:
            """Primary event stream: JSON events from step_engine over TCP."""
            nonlocal post_agents, engine_globals
            if _tcp_reader is None:
                return
            try:
                while True:
                    try:
                        raw = await _tcp_reader.readline()
                    except Exception:
                        break
                    if not raw:
                        break
                    line = raw.decode().rstrip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        event["run"] = run_name
                        if event.get("type") == "post_processing_done":
                            post_agents = event.pop("agents", [])
                            engine_killed_agents = event.pop("killed_agents", [])
                            engine_globals = event.pop("globals", {})
                        # Store agent_log and agent_status lines server-side for replay
                        evt_type = event.get("type")
                        rank = event.get("agent_rank")
                        if rank is not None and evt_type == "worker_pid":
                            run.agent_pids[rank] = event.get("pid")
                            continue  # internal event — don't broadcast
                        if rank is not None and evt_type == "agent_completed":
                            run.agent_pids.pop(rank, None)
                        if rank is not None and evt_type == "agent_failed":
                            failed_ranks.add(rank)
                            run.agent_pids.pop(rank, None)
                        if rank is not None:
                            if evt_type == "agent_log":
                                stream = event.get("stream", "stdout")
                                text   = event.get("text", "")
                                prefix = "" if stream == "stdout" else "[stderr] "
                                run.agent_logs.setdefault(rank, []).append(prefix + text)
                            elif evt_type == "agent_status":
                                status = event.get("status", "")
                                if status:
                                    run.agent_logs.setdefault(rank, []).append(f"[status] {status}")
                        await _bcast_queue.put(event)
                    except json.JSONDecodeError:
                        await _bcast_queue.put({"type": "log_line", "run": run_name, "text": line})
            finally:
                # Close our TCP writer so step_engine's keep-alive reader.read() gets
                # EOF, unblocking _ui_server.wait_closed() and letting the process exit.
                if _tcp_writer is not None and not _tcp_writer.is_closing():
                    try:
                        _tcp_writer.close()
                    except Exception:
                        pass

        async def read_stdout() -> None:
            """Capture remaining stdout for debugging (no event parsing in TCP mode)."""
            async for raw in proc.stdout:  # type: ignore[union-attr]
                line = raw.decode().rstrip()
                if line:
                    await _bcast_queue.put({"type": "log_line", "run": run_name, "text": f"[stdout] {line}"})

        async def read_stderr() -> None:
            assert proc.stderr
            async for raw in proc.stderr:
                line = raw.decode().rstrip()
                if line:
                    proc_stderr_lines.append(line)
                    # Broadcast live to all agents currently known to be running
                    ranks = list(run.agent_logs.keys()) or [0]
                    for rank in ranks:
                        await broadcast(run_name, {
                            "type": "agent_log", "run": run_name,
                            "agent_rank": rank, "stream": "stderr", "text": line,
                        })

        try:
            await asyncio.gather(read_tcp(), read_stdout(), read_stderr())
            await proc.wait()
        except Exception as exc:
            # Stream read failed (e.g. line exceeded buffer limit).  The process
            # may still be running; kill it to avoid a zombie.
            try:
                proc.kill()
            except Exception:
                pass
            run.running_pid = None
            run.running_operator = None
            await _bcast_queue.put(None)  # stop bcast_task before re-raising
            await bcast_task
            await _broadcast_and_log(run_name, {
                "type": "step_failed",
                "run": run_name,
                "operator_name": operator_name,
                "error": f"Stream read error: {exc}",
            })
            return

        # Drain remaining broadcasts before moving on
        await _bcast_queue.put(None)  # sentinel
        await bcast_task

        run.running_pid = None
        run.running_operator = None

        if proc.returncode == 0 and post_agents:
            # If any agents were killed by the user this step, move them from
            # active to the killed list in engine_state.pt (so next step skips them).
            if run.killed_ranks:
                states_dir = RUNS_DIR / run_name / "engine_states"
                pts = sorted(states_dir.glob("*.pt")) if states_dir.exists() else []
                state_file = pts[-1] if pts else None
                if state_file:
                    try:
                        import torch as _torch
                        data = _torch.load(state_file, weights_only=False)
                        # Handle both engine format and UI snapshot format
                        agents_key = "full_agents" if "full_agents" in data and "agents" not in data else "agents"
                        newly_killed, still_active = [], []
                        for a in data.get(agents_key, []):
                            if a.get("agent_rank") in run.killed_ranks:
                                a["agent_killed"] = True
                                newly_killed.append(a)
                            else:
                                still_active.append(a)
                        data[agents_key] = still_active + newly_killed
                        if agents_key == "agents":
                            data["killed_agents"] = data.get("killed_agents", []) + newly_killed
                            data[agents_key] = still_active
                        _torch.save(data, state_file)
                    except Exception as _patch_err:
                        print(f"[subprocess_manager] Failed to patch engine state: {_patch_err}", file=sys.stderr)

            # Identify newly-killed agents using pre-step state (by rank → uid),
            # so we are immune to any reindexing the engine did after removing the
            # killed subprocess.  Build the correct killed-agent dicts from
            # pre_agents, then strip any stale entries from post_agents.
            if run.killed_ranks:
                pre_by_rank = {a.get("agent_rank"): a for a in pre_agents}
                killed_uids: set = set()
                newly_killed_entries: list = []
                for rank in run.killed_ranks:
                    orig = pre_by_rank.get(rank)
                    if orig:
                        ka = dict(orig)
                        ka["agent_killed"] = True
                        newly_killed_entries.append(ka)
                        uid = orig.get("unique_id")
                        if uid:
                            killed_uids.add(uid)
                # Remove killed agents from post_agents.  If we have uids (the
                # normal case), match by uid — safe even after reindexing.
                # Otherwise fall back to rank, which may be wrong but is all we have.
                if killed_uids:
                    post_agents = [a for a in post_agents if a.get("unique_id") not in killed_uids]
                else:
                    post_agents = [a for a in post_agents if a.get("agent_rank") not in run.killed_ranks]
                post_agents.extend(newly_killed_entries)

            # Preserve failed agents from pre-step state with agent_failed=True,
            # immune to engine reindexing (same pattern as killed agents).
            if failed_ranks:
                pre_by_rank = {a.get("agent_rank"): a for a in pre_agents}
                failed_uids: set = set()
                newly_failed_entries: list = []
                for rank in failed_ranks:
                    orig = pre_by_rank.get(rank)
                    if orig:
                        fa = dict(orig)
                        fa["agent_failed"] = True
                        newly_failed_entries.append(fa)
                        uid = orig.get("unique_id")
                        if uid:
                            failed_uids.add(uid)
                if failed_uids:
                    post_agents = [a for a in post_agents if a.get("unique_id") not in failed_uids]
                else:
                    post_agents = [a for a in post_agents if a.get("agent_rank") not in failed_ranks]
                post_agents.extend(newly_failed_entries)

            post_agents.sort(key=lambda a: (1 if (a.get("agent_killed") or a.get("agent_failed")) else 0, a.get("agent_rank", 0)))

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
            run.step_log = []  # clear so reconnecting clients don't see phantom "running"
            await broadcast(run_name, {
                "type": "step_completed",
                "run": run_name,
                "operator_name": operator_name,
                "step_num": step_num,
                "engine_state_uid": record.uid,
                "record": state_manager.record_to_summary(record),
            })
        elif proc.returncode == 0:
            # Step engine exited cleanly but reported zero surviving agents
            # (e.g. kill_failed=True removed every agent that failed).
            # We must still notify the UI so it doesn't hang in "Running" state.
            elim_msg = "All agents were eliminated — kill_failed removed every agent that failed this step."
            if proc_stderr_lines:
                elim_msg = "\n".join(proc_stderr_lines).strip() + "\n\n" + elim_msg
            await _broadcast_and_log(run_name, {
                "type": "step_failed",
                "run": run_name,
                "operator_name": operator_name,
                "error": elim_msg,
            })
        else:
            # proc.returncode != 0
            # Persist proc-level stderr into every agent's log buffer for replay.
            # (They were already broadcast live in read_stderr above.)
            if proc_stderr_lines:
                ranks = list(run.agent_logs.keys()) or [0]
                for rank in ranks:
                    buf = run.agent_logs.setdefault(rank, [])
                    buf.append("[stderr] --- process stderr ---")
                    buf.extend(f"[stderr] {l}" for l in proc_stderr_lines)
            error_text = "\n".join(proc_stderr_lines).strip() if proc_stderr_lines \
                else f"Process exited with code {proc.returncode}"
            await _broadcast_and_log(run_name, {
                "type": "step_failed",
                "run": run_name,
                "operator_name": operator_name,
                "error": error_text,
            })

    except Exception as exc:
        run.running_pid = None
        run.running_operator = None
        await _broadcast_and_log(run_name, {
            "type": "step_failed",
            "run": run_name,
            "operator_name": operator_name,
            "error": str(exc),
        })
