"""Worker-side IPC channel (TCP to engine process).

Configured at worker startup when BSA_ENGINE_TCP_PORT env var is set.
Falls back to stdout/stdin in standalone / direct-invocation mode so
existing behaviour is fully preserved without the engine.
"""
from __future__ import annotations

import asyncio
import json
import sys
from typing import Optional

_reader: Optional[asyncio.StreamReader] = None
_writer: Optional[asyncio.StreamWriter] = None


def configure(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    global _reader, _writer
    _reader = reader
    _writer = writer


def is_connected() -> bool:
    return _writer is not None and not _writer.is_closing()


def emit(event: dict) -> None:
    """Non-blocking fire-and-forget event.  TCP if connected, stdout otherwise."""
    if is_connected():
        # write() buffers immediately; asyncio transport flushes on next loop tick.
        _writer.write((json.dumps(event) + "\n").encode())  # type: ignore[union-attr]
    else:
        print(json.dumps(event), flush=True, file=sys.stdout)


async def drain() -> None:
    """Flush any pending TCP writes."""
    if is_connected():
        await _writer.drain()  # type: ignore[union-attr]


async def request_tool_slot(agent_rank: int) -> None:
    """Signal engine we need a tool slot; suspend until the engine grants it."""
    if is_connected():
        emit({"type": "tool_slot_request", "agent_rank": agent_rank})
        await drain()
        assert _reader is not None
        await _reader.readline()   # consume {"type":"tool_slot_grant"} response
        return
    # Fallback: legacy stdin handshake (standalone mode)
    if sys.stdin.isatty():
        return
    print(json.dumps({"type": "tool_slot_request", "agent_rank": agent_rank}), flush=True)
    await asyncio.to_thread(sys.stdin.readline)


def release_tool_slot(agent_rank: int) -> None:
    """Notify engine that the tool slot is no longer needed."""
    if is_connected():
        emit({"type": "tool_slot_release", "agent_rank": agent_rank})
    elif not sys.stdin.isatty():
        print(json.dumps({"type": "tool_slot_release", "agent_rank": agent_rank}), flush=True)


async def close() -> None:
    """Drain and close the TCP connection cleanly."""
    global _reader, _writer
    if _writer is not None:
        try:
            await _writer.drain()
            _writer.close()
            await _writer.wait_closed()
        except Exception:
            pass
    _reader = None
    _writer = None
