"""WebSocket connection registry and broadcast helpers."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import state_manager

router = APIRouter()

# run_name -> set of active connections
_connections: dict[str, set[WebSocket]] = defaultdict(set)


@router.websocket("/ws/{run_name}")
async def ws_endpoint(websocket: WebSocket, run_name: str) -> None:
    await websocket.accept()
    _connections[run_name].add(websocket)

    # Replay current execution state to the newly connected client
    if state_manager.has_run(run_name):
        run = state_manager.get_run(run_name)
        # Send current queue so the operator panel and tree show queued ops
        await websocket.send_json({
            "type": "queue_updated",
            "run": run_name,
            "queue": list(run.queue),
        })
        # Replay buffered step events (step_started, agent_log, agent_completed, etc.)
        for event in list(run.step_log):
            try:
                await websocket.send_json(event)
            except Exception:
                break

    try:
        while True:
            await websocket.receive_text()  # keep-alive; discard client messages
    except WebSocketDisconnect:
        _connections[run_name].discard(websocket)


async def broadcast(run_name: str, message: dict[str, Any]) -> None:
    dead: set[WebSocket] = set()
    for ws in list(_connections[run_name]):
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)
    _connections[run_name] -= dead


async def broadcast_all(message: dict[str, Any]) -> None:
    for run_name in list(_connections.keys()):
        await broadcast(run_name, message)
