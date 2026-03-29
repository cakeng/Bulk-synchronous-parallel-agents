"""BSA UI Server — FastAPI application entry point.

Usage:
    cd /data/js_park/bsa
    python ui/server.py        # or: python -m ui.server
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Allow running as a script (python3 ui/server.py) by re-execing as a module.
if __name__ == "__main__" and __package__ is None:
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "ui.server"],
        cwd=str(root),
    )
    sys.exit(result.returncode)

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import state_manager
from .routes import runs, operators, engine, websocket

app = FastAPI(title="BSA UI")

# API routes
app.include_router(runs.router)
app.include_router(operators.router)
app.include_router(engine.router)
app.include_router(websocket.router)

# Serve frontend static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


@app.on_event("startup")
async def startup() -> None:
    state_manager.load_all_runs()


if __name__ == "__main__":
    uvicorn.run("ui.server:app", host="127.0.0.1", port=18001, reload=True)
