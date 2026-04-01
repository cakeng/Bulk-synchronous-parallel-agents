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


class CopyRunBody(BaseModel):
    new_name: str


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


@router.post("/{run_name}/copy")
async def copy_run(run_name: str, body: CopyRunBody) -> dict:
    if not state_manager.has_run(run_name):
        raise HTTPException(404, f"Run '{run_name}' not found")
    new_name = body.new_name.strip()
    if not new_name or "/" in new_name:
        raise HTTPException(400, "Invalid run name")
    if state_manager.has_run(new_name):
        raise HTTPException(409, f"Run '{new_name}' already exists")
    src_dir = RUNS_DIR / run_name
    dst_dir = RUNS_DIR / new_name
    shutil.copytree(src_dir, dst_dir)
    state_manager.create_run(new_name)
    state_manager.reload_run(new_name)
    await broadcast_all({"type": "runs_updated", "runs": state_manager.get_all_run_names()})
    return {"name": new_name}


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


@router.post("/{run_name}/kill_agent/{agent_rank}")
async def kill_agent(run_name: str, agent_rank: int) -> dict:
    if not state_manager.has_run(run_name):
        raise HTTPException(404, f"Run '{run_name}' not found")
    found = await subprocess_manager.kill_agent(run_name, agent_rank)
    if not found:
        raise HTTPException(404, f"No running worker found for agent {agent_rank}")
    return {"killed_agent": agent_rank}


@router.post("/{run_name}/halfbake")
async def halfbake_step(run_name: str) -> dict:
    """Kill all still-running agent workers; already-completed agents form the step result."""
    if not state_manager.has_run(run_name):
        raise HTTPException(404, f"Run '{run_name}' not found")
    run = state_manager.get_run(run_name)
    ranks = list(run.agent_pids.keys())
    if not ranks:
        raise HTTPException(404, "No running agent workers found")
    for rank in ranks:
        await subprocess_manager.kill_agent(run_name, rank)
    return {"halfbaked_ranks": ranks}


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
    # Delete workspace directories
    workspaces_dir = RUNS_DIR / run_name / "workspaces"
    if workspaces_dir.exists():
        shutil.rmtree(workspaces_dir)
    # Remove output symlinks (targets are gone)
    outputs_dir = RUNS_DIR / run_name / "outputs"
    if outputs_dir.exists():
        for link in outputs_dir.glob("agent_*"):
            if link.is_symlink():
                link.unlink(missing_ok=True)
    state_manager.clear_run(run_name)
    await broadcast_all({"type": "runs_updated", "runs": state_manager.get_all_run_names()})
    return {"cleared": run_name}
