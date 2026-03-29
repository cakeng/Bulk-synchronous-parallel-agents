"""Operator CRUD, reorder, and execution endpoints."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import state_manager, subprocess_manager
from ..routes.websocket import broadcast
from src import op_codegen

router = APIRouter(prefix="/api/runs/{run_name}/operators")
RUNS_DIR = Path("runs")


class CreateOperatorBody(BaseModel):
    name: str
    op_type: str = "base"


class UpdateOperatorBody(BaseModel):
    body: str


class ReorderBody(BaseModel):
    names: list[str]


class CopyOperatorBody(BaseModel):
    new_name: str


class RenameOperatorBody(BaseModel):
    new_name: str


def _run_or_404(run_name: str):
    if not state_manager.has_run(run_name):
        raise HTTPException(404, f"Run '{run_name}' not found")
    return state_manager.get_run(run_name)


def _op_path(run_name: str, op_name: str) -> Path:
    return RUNS_DIR / run_name / "operators" / op_name


@router.get("")
async def list_operators(run_name: str) -> dict:
    run = _run_or_404(run_name)
    ops = []
    for spec in run.operator_list:
        path = _op_path(run_name, spec.name)
        if path.exists():
            code = path.read_text()
            body = op_codegen.extract_body(code)
            op_type = op_codegen.detect_op_type(code)
        else:
            body = ""
            op_type = "base"
        ops.append({"name": spec.name, "body": body, "op_type": op_type})
    return {"operators": ops}


@router.post("")
async def create_operator(run_name: str, body: CreateOperatorBody) -> dict:
    run = _run_or_404(run_name)
    name = body.name.strip()
    if not name.endswith(".py"):
        name += ".py"
    if any(op.name == name for op in run.operator_list):
        raise HTTPException(409, f"Operator '{name}' already exists")

    op_body = op_codegen.DEFAULT_BODIES.get(body.op_type, op_codegen.DEFAULT_BODIES["base"])
    code = op_codegen.generate_full_code(name, op_body, body.op_type)
    _op_path(run_name, name).write_text(code)

    run.operator_list.append(state_manager.OperatorSpec(name=name))
    state_manager.save_operator_order(run_name)
    return {"name": name, "body": op_body}


@router.put("/{op_name}")
async def update_operator(run_name: str, op_name: str, body: UpdateOperatorBody) -> dict:
    _run_or_404(run_name)
    path = _op_path(run_name, op_name)
    existing = path.read_text() if path.exists() else ""
    op_type = op_codegen.detect_op_type(existing)
    path.write_text(op_codegen.generate_full_code(op_name, body.body, op_type))
    return {"name": op_name}


@router.delete("/{op_name}")
async def delete_operator(run_name: str, op_name: str) -> dict:
    run = _run_or_404(run_name)

    if run.running_operator == op_name:
        await subprocess_manager.kill_current(run_name)
    await subprocess_manager.remove_from_queue(run_name, op_name)

    run.operator_list = [op for op in run.operator_list if op.name != op_name]
    state_manager.save_operator_order(run_name)
    _op_path(run_name, op_name).unlink(missing_ok=True)
    return {"deleted": op_name}


@router.post("/{op_name}/copy")
async def copy_operator(run_name: str, op_name: str, body: CopyOperatorBody) -> dict:
    run = _run_or_404(run_name)
    src = _op_path(run_name, op_name)
    if not src.exists():
        raise HTTPException(404, f"Operator '{op_name}' not found")

    new_name = body.new_name.strip()
    if not new_name.endswith(".py"):
        new_name += ".py"

    src_code = src.read_text()
    op_body  = op_codegen.extract_body(src_code)
    op_type  = op_codegen.detect_op_type(src_code)
    new_code = op_codegen.generate_full_code(new_name, op_body, op_type)

    dst = _op_path(run_name, new_name)
    dst.write_text(new_code)
    run.operator_list.append(state_manager.OperatorSpec(name=new_name))
    state_manager.save_operator_order(run_name)
    return {"name": new_name, "body": op_body}


@router.post("/{op_name}/rename")
async def rename_operator(run_name: str, op_name: str, body: RenameOperatorBody) -> dict:
    run = _run_or_404(run_name)
    new_name = body.new_name.strip()
    if not new_name.endswith(".py"):
        new_name += ".py"
    if new_name == op_name:
        src = _op_path(run_name, op_name)
        return {"name": op_name, "body": op_codegen.extract_body(src.read_text() if src.exists() else "")}
    if any(op.name == new_name for op in run.operator_list):
        raise HTTPException(409, f"Operator '{new_name}' already exists")

    src = _op_path(run_name, op_name)
    if not src.exists():
        raise HTTPException(404, f"Operator '{op_name}' not found")

    src_code = src.read_text()
    op_body  = op_codegen.extract_body(src_code)
    op_type  = op_codegen.detect_op_type(src_code)
    new_code = op_codegen.generate_full_code(new_name, op_body, op_type)

    _op_path(run_name, new_name).write_text(new_code)
    src.unlink()

    for op in run.operator_list:
        if op.name == op_name:
            op.name = new_name
            break
    state_manager.save_operator_order(run_name)
    return {"name": new_name, "body": op_body}


@router.post("/reorder")
async def reorder_operators(run_name: str, body: ReorderBody) -> dict:
    run = _run_or_404(run_name)
    name_set = {op.name for op in run.operator_list}
    run.operator_list = [state_manager.OperatorSpec(name=n)
                         for n in body.names if n in name_set]
    state_manager.save_operator_order(run_name)
    return {"order": [op.name for op in run.operator_list]}


@router.post("/{op_name}/run")
async def run_operator(run_name: str, op_name: str) -> dict:
    run = _run_or_404(run_name)
    if not any(op.name == op_name for op in run.operator_list):
        raise HTTPException(404, f"Operator '{op_name}' not found in list")
    await subprocess_manager.enqueue(run_name, op_name)
    return {"queued": op_name}
