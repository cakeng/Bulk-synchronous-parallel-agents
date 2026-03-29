"""WebSocket connection registry and broadcast helpers."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

# run_name -> set of active connections
_connections: dict[str, set[WebSocket]] = defaultdict(set)


@router.websocket("/ws/{run_name}")
async def ws_endpoint(websocket: WebSocket, run_name: str) -> None:
    await websocket.accept()
    _connections[run_name].add(websocket)
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
