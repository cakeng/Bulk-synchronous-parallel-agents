"""Run CRUD endpoints."""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import state_manager
from ..routes.websocket import broadcast_all
from .. import subprocess_manager

router = APIRouter(prefix="/api/runs")
RUNS_DIR = Path("runs")


class CreateRunBody(BaseModel):
    name: str


@router.get("")
async def list_runs() -> dict:
    return {"runs": state_manager.get_all_run_names()}


@router.post("")
async def create_run(body: CreateRunBody) -> dict:
    name = body.name.strip()
    if not name or "/" in name:
        raise HTTPException(400, "Invalid run name")
    if state_manager.has_run(name):
        raise HTTPException(409, f"Run '{name}' already exists")
    state_manager.create_run(name)
    await broadcast_all({"type": "runs_updated", "runs": state_manager.get_all_run_names()})
    return {"name": name}


@router.delete("/{run_name}")
async def delete_run(run_name: str) -> dict:
    if not state_manager.has_run(run_name):
        raise HTTPException(404, f"Run '{run_name}' not found")
    await subprocess_manager.kill_current(run_name)
    state_manager.delete_run(run_name)
    run_dir = RUNS_DIR / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    await broadcast_all({"type": "runs_updated", "runs": state_manager.get_all_run_names()})
    return {"deleted": run_name}


@router.get("/{run_name}/status")
async def get_run_status(run_name: str) -> dict:
    if not state_manager.has_run(run_name):
        raise HTTPException(404, f"Run '{run_name}' not found")
    run = state_manager.get_run(run_name)
    return {
        "running_operator": run.running_operator,
        "queue": list(run.queue),
        "step_log": list(run.step_log),
        "agent_logs": {str(k): v for k, v in run.agent_logs.items()},
    }


@router.post("/{run_name}/kill")
async def kill_run(run_name: str) -> dict:
    if not state_manager.has_run(run_name):
        raise HTTPException(404, f"Run '{run_name}' not found")
    await subprocess_manager.kill_current(run_name)
    return {"killed": run_name}


@router.post("/{run_name}/clear")
async def clear_run(run_name: str) -> dict:
    if not state_manager.has_run(run_name):
        raise HTTPException(404, f"Run '{run_name}' not found")
    await subprocess_manager.kill_current(run_name)
    # Delete all engine state files
    states_dir = RUNS_DIR / run_name / "engine_states"
    if states_dir.exists():
        for f in states_dir.glob("*.pt"):
            f.unlink(missing_ok=True)
    # Also remove legacy engine_state.pt
    legacy = RUNS_DIR / run_name / "engine_state.pt"
    legacy.unlink(missing_ok=True)
    state_manager.clear_run(run_name)
    await broadcast_all({"type": "runs_updated", "runs": state_manager.get_all_run_names()})
    return {"cleared": run_name}
