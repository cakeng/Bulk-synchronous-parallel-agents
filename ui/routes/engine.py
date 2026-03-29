"""Engine state read endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import state_manager

router = APIRouter(prefix="/api/runs/{run_name}/engine_states")


def _run_or_404(run_name: str):
    if not state_manager.has_run(run_name):
        raise HTTPException(404, f"Run '{run_name}' not found")
    return state_manager.get_run(run_name)


@router.get("")
async def list_engine_states(run_name: str) -> dict:
    run = _run_or_404(run_name)
    return {
        "engine_states": [state_manager.record_to_summary(r)
                          for r in run.engine_state_list]
    }


@router.get("/{uid}")
async def get_engine_state(run_name: str, uid: str) -> dict:
    run = _run_or_404(run_name)
    record = next((r for r in run.engine_state_list if r.uid == uid), None)
    if record is None:
        raise HTTPException(404, f"Engine state '{uid}' not found")
    full = state_manager.load_full_engine_state(record.file_path)
    return {
        "uid": record.uid,
        "parent_uid": record.parent_uid,
        "step_num": record.step_num,
        "operator_name": record.operator_name,
        "operator_code": record.operator_code,
        "timestamp": record.timestamp,
        "globals": full.get("globals", {}),
        "agents": full.get("full_agents", full.get("post_agents", [])),
    }


@router.delete("/from/{from_index}")
async def delete_from(run_name: str, from_index: int) -> dict:
    """Remove all engine states from from_index to end (0-based)."""
    run = _run_or_404(run_name)
    if from_index < 0 or from_index > len(run.engine_state_list):
        raise HTTPException(400, "Invalid index")
    removed = state_manager.remove_engine_states_from_index(run_name, from_index)
    return {"removed_uids": removed}
