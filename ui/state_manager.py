"""In-memory state management for the BSA UI server."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from src import op_codegen

RUNS_DIR = Path("runs")


@dataclass
class EngineStateRecord:
    uid: str
    parent_uid: str | None
    step_num: int
    operator_name: str
    operator_type: str
    operator_code: str
    timestamp: float
    file_path: str          # absolute path to .pt file
    # Structural summary for the tree (lightweight)
    pre_agents: list[dict]  # agents before this step
    post_agents: list[dict] # agents after this step (result)


@dataclass
class OperatorSpec:
    name: str           # filename e.g. "step1.py"
    kill_failed: bool = True


@dataclass
class RunState:
    name: str
    engine_state_list: list[EngineStateRecord] = field(default_factory=list)
    operator_list: list[OperatorSpec] = field(default_factory=list)
    queue: list[str] = field(default_factory=list)
    running_operator: str | None = None
    running_pid: int | None = None
    # Buffered events from the current/last step for reconnecting clients
    step_log: list[dict] = field(default_factory=list)
    # Per-agent stdout/stderr lines for the current step (rank -> [lines])
    agent_logs: dict[int, list[str]] = field(default_factory=dict)


_runs: dict[str, RunState] = {}


# ---------------------------------------------------------------------------
# Run registry
# ---------------------------------------------------------------------------

def get_all_run_names() -> list[str]:
    return sorted(_runs.keys())


def get_run(run_name: str) -> RunState:
    return _runs[run_name]


def has_run(run_name: str) -> bool:
    return run_name in _runs


def create_run(run_name: str) -> RunState:
    run_dir = RUNS_DIR / run_name
    (run_dir / "operators").mkdir(parents=True, exist_ok=True)
    (run_dir / "engine_states").mkdir(parents=True, exist_ok=True)
    state = RunState(name=run_name)
    _runs[run_name] = state
    return state


def delete_run(run_name: str) -> None:
    _runs.pop(run_name, None)


def clear_run(run_name: str) -> None:
    """Remove all engine state records from memory (disk deletion done by caller)."""
    run = _runs.get(run_name)
    if run:
        run.engine_state_list.clear()
        run.queue.clear()
        run.running_operator = None
        run.running_pid = None


# ---------------------------------------------------------------------------
# Startup loader
# ---------------------------------------------------------------------------

def load_all_runs() -> None:
    """Scan runs/ on server startup and populate in-memory state."""
    if not RUNS_DIR.exists():
        return
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name
        (run_dir / "operators").mkdir(exist_ok=True)
        (run_dir / "engine_states").mkdir(exist_ok=True)

        state = RunState(name=run_name)
        state.engine_state_list = _load_engine_states(run_name)
        state.operator_list = _load_operator_list(run_name)
        _runs[run_name] = state


def _load_engine_states(run_name: str) -> list[EngineStateRecord]:
    states_dir = RUNS_DIR / run_name / "engine_states"
    records: list[EngineStateRecord] = []
    for pt_file in sorted(states_dir.glob("*.pt")):
        try:
            data = torch.load(pt_file, weights_only=False)
            if not isinstance(data, dict) or "uid" not in data:
                continue
            records.append(EngineStateRecord(
                uid=data["uid"],
                parent_uid=data.get("parent_uid"),
                step_num=data["step_num"],
                operator_name=data["operator_name"],
                operator_type=data.get("operator_type", op_codegen.detect_op_type(data.get("operator_code", ""))),
                operator_code=data.get("operator_code", ""),
                timestamp=data["timestamp"],
                file_path=str(pt_file.resolve()),
                pre_agents=data.get("pre_agents", []),
                post_agents=data.get("post_agents", []),
            ))
        except Exception:
            pass
    records.sort(key=lambda r: (r.step_num, r.timestamp))
    return records


def _load_operator_list(run_name: str) -> list[OperatorSpec]:
    run_dir = RUNS_DIR / run_name
    order_file = run_dir / "operator_order.json"
    if order_file.exists():
        try:
            entries = json.loads(order_file.read_text())
            specs = []
            for entry in entries:
                if isinstance(entry, str):
                    name, kf = entry, False          # legacy format
                else:
                    name, kf = entry["name"], entry.get("kill_failed", False)
                if (run_dir / "operators" / name).exists():
                    specs.append(OperatorSpec(name=name, kill_failed=kf))
            return specs
        except Exception:
            pass
    return [OperatorSpec(name=f.name)
            for f in sorted((run_dir / "operators").glob("*.py"))]


# ---------------------------------------------------------------------------
# Operator management
# ---------------------------------------------------------------------------

def save_operator_order(run_name: str) -> None:
    run = _runs[run_name]
    order_file = RUNS_DIR / run_name / "operator_order.json"
    order_file.write_text(json.dumps(
        [{"name": op.name, "kill_failed": op.kill_failed} for op in run.operator_list]
    ))


# ---------------------------------------------------------------------------
# Engine state records
# ---------------------------------------------------------------------------

def add_engine_state_record(
    run_name: str,
    step_num: int,
    operator_name: str,
    operator_code: str,
    timestamp: float,
    pre_agents: list[dict],
    post_agents: list[dict],
    engine_globals: dict,
) -> EngineStateRecord:
    run = _runs[run_name]
    parent_uid = run.engine_state_list[-1].uid if run.engine_state_list else None
    uid = uuid.uuid4().hex[:8]
    operator_type = op_codegen.detect_op_type(operator_code)

    filename = f"{run_name}_{step_num:04d}_engine_state_{int(timestamp)}.pt"
    file_path = (RUNS_DIR / run_name / "engine_states" / filename).resolve()

    record = EngineStateRecord(
        uid=uid,
        parent_uid=parent_uid,
        step_num=step_num,
        operator_name=operator_name,
        operator_type=operator_type,
        operator_code=operator_code,
        timestamp=timestamp,
        file_path=str(file_path),
        pre_agents=_slim_agents(pre_agents),
        post_agents=_slim_agents(post_agents),
    )

    torch.save({
        "uid": uid,
        "parent_uid": parent_uid,
        "step_num": step_num,
        "operator_name": operator_name,
        "operator_type": operator_type,
        "operator_code": operator_code,
        "timestamp": timestamp,
        "pre_agents": _slim_agents(pre_agents),
        "post_agents": _slim_agents(post_agents),
        "full_agents": post_agents,
        "globals": engine_globals,
    }, file_path)

    run.engine_state_list.append(record)
    return record


def _slim_agents(agents: list[dict]) -> list[dict]:
    """Keep only structural fields for tree rendering."""
    slim_keys = {"agent_rank", "unique_id", "parent_id", "fork_rank"}
    result = []
    for a in agents:
        slim = {k: v for k, v in a.items() if k in slim_keys}
        if "shuffle_output" in a and isinstance(a["shuffle_output"], dict):
            slim["shuffle_sources"] = sorted(a["shuffle_output"].keys())
        result.append(slim)
    return result


def load_full_engine_state(file_path: str) -> dict:
    return torch.load(file_path, weights_only=False)


def remove_engine_states_from_index(run_name: str, from_index: int) -> list[str]:
    """Remove engine states from index onwards. Returns list of removed uids."""
    run = _runs[run_name]
    removed = run.engine_state_list[from_index:]
    run.engine_state_list = run.engine_state_list[:from_index]
    for record in removed:
        try:
            Path(record.file_path).unlink(missing_ok=True)
        except Exception:
            pass
    return [r.uid for r in removed]


def record_to_summary(r: EngineStateRecord) -> dict:
    return {
        "uid": r.uid,
        "parent_uid": r.parent_uid,
        "step_num": r.step_num,
        "operator_name": r.operator_name,
        "operator_type": r.operator_type,
        "timestamp": r.timestamp,
        "pre_agents": r.pre_agents,
        "post_agents": r.post_agents,
    }
