#!/usr/bin/env python3
"""
Code execution tool — run Python code in a UV virtual environment within the workspace.

Usage:
  python tools/code.py <workspace_dir> '{"action": "run_code", "code": "print(1+1)"}'
  python tools/code.py <workspace_dir> '{"action": "run_file", "path": "script.py"}'
"""

TOOL_NAME = "code"

TOOL_PROMPTS = {
    "full": """\
--- Tool: code ---
Execute Python code in a UV virtual environment (.venv) inside your workspace.
The working directory for executed code is your workspace directory.

  run_code      Execute a Python source string.
    {"tool_call": {"tool": "code", "action": "run_code", "code": "<python source>", "timeout": 60}}

  run_file      Execute a .py script from your workspace.
    {"tool_call": {"tool": "code", "action": "run_file", "path": "<filename.py>", "args": [], "timeout": 60}}

  save_and_run  Save code to a file in your workspace and immediately execute it.
    {"tool_call": {"tool": "code", "action": "save_and_run", "path": "<filename.py>", "code": "<python source>", "timeout": 60}}

  pip_install   Install packages into the .venv using uv pip install.
    {"tool_call": {"tool": "code", "action": "pip_install", "packages": "numpy pandas>=2.0", "timeout": 120}}

All actions return: {"stdout": "...", "stderr": "...", "exit_code": 0, "timed_out": false}
timeout is in seconds (default 30). Set explicitly for long-running code; retry with a longer value if it times out.
""",
}

TOOL_SCHEMAS = {
    "full": [
        {
            "type": "function",
            "function": {
                "name": "code__run_code",
                "description": "Execute a Python source string inside the UV virtual environment. Working directory is the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python source code to execute"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "code__run_file",
                "description": "Execute a .py script from the workspace inside the UV virtual environment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to .py script relative to workspace"},
                        "args": {"type": "array", "items": {"type": "string"}, "description": "Command-line arguments"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "code__save_and_run",
                "description": "Save Python source code to the workspace and immediately execute it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Destination path relative to workspace (e.g. 'script.py')"},
                        "code": {"type": "string", "description": "Python source code to save and run"},
                        "args": {"type": "array", "items": {"type": "string"}, "description": "Command-line arguments"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
                    },
                    "required": ["path", "code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "code__pip_install",
                "description": "Install Python packages into the UV virtual environment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "packages": {"type": "string", "description": "Space-separated package specs (e.g. 'numpy pandas>=2.0')"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
                    },
                    "required": ["packages"],
                },
            },
        },
    ],
}

import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_TIMEOUT = 30


def _venv_python(workspace_dir: Path) -> Path:
    return workspace_dir / ".venv" / "bin" / "python"


def _ensure_venv(workspace_dir: Path) -> dict | None:
    """Create .venv in workspace_dir using uv if it doesn't exist."""
    venv_python = _venv_python(workspace_dir)
    if venv_python.exists():
        return None
    workspace_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ["uv", "venv", str(workspace_dir / ".venv")],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(workspace_dir),
        )
        if result.returncode != 0:
            return {"success": False, "error": f"Failed to create venv: {result.stderr.strip()}"}
    except FileNotFoundError:
        return {"success": False, "error": "uv executable not found on PATH"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "uv venv creation timed out"}
    return None


def _run(cmd: list, timeout: int | None, workspace_dir: Path, used_default_timeout: bool = False) -> dict:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(workspace_dir),
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": proc.returncode,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as e:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        remaining_out, remaining_err = proc.communicate()
        note = (
            f"Timed out after {timeout}s (default timeout — set an explicit timeout if your code needs longer)."
            if used_default_timeout else
            f"Timed out after {timeout}s."
        )
        return {
            "stdout": (e.stdout or "") + (remaining_out or ""),
            "stderr": (e.stderr or "") + (remaining_err or ""),
            "exit_code": None,
            "timed_out": True,
            "note": note,
        }


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def run_file(path: str, workspace_dir: Path, args: list = None, timeout: int = None) -> dict:
    err = _ensure_venv(workspace_dir)
    if err:
        return err
    p = (workspace_dir / path).resolve()
    if not str(p).startswith(str(workspace_dir.resolve())):
        raise ValueError(f"Path must be within workspace, got: {path!r}")
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    cmd = [str(_venv_python(workspace_dir)), str(p)] + (args or [])
    used_default = timeout is None
    result = _run(cmd, timeout or DEFAULT_TIMEOUT, workspace_dir, used_default)
    result["path"] = str(p.relative_to(workspace_dir.resolve()))
    return result


def run_code(code: str, workspace_dir: Path, timeout: int = None) -> dict:
    err = _ensure_venv(workspace_dir)
    if err:
        return err
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="bsa_run_") as f:
        f.write(code)
        tmp = Path(f.name)
    used_default = timeout is None
    try:
        return _run([str(_venv_python(workspace_dir)), str(tmp)], timeout or DEFAULT_TIMEOUT, workspace_dir, used_default)
    finally:
        tmp.unlink(missing_ok=True)


def save_and_run(path: str, code: str, workspace_dir: Path, args: list = None, timeout: int = None) -> dict:
    err = _ensure_venv(workspace_dir)
    if err:
        return err
    p = (workspace_dir / path).resolve()
    if not str(p).startswith(str(workspace_dir.resolve())):
        raise ValueError(f"Path must be within workspace, got: {path!r}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(code)
    cmd = [str(_venv_python(workspace_dir)), str(p)] + (args or [])
    used_default = timeout is None
    result = _run(cmd, timeout or DEFAULT_TIMEOUT, workspace_dir, used_default)
    result["path"] = str(p.relative_to(workspace_dir.resolve()))
    return result


def pip_install(packages: str, workspace_dir: Path, timeout: int = None) -> dict:
    err = _ensure_venv(workspace_dir)
    if err:
        return err
    try:
        pkg_args = shlex.split(packages)
    except ValueError as e:
        return {"success": False, "error": f"Could not parse packages: {e}"}
    if not pkg_args:
        return {"success": False, "error": "packages must not be empty"}
    cmd = ["uv", "pip", "install", "--python", str(_venv_python(workspace_dir))] + pkg_args
    used_default = timeout is None
    return _run(cmd, timeout or DEFAULT_TIMEOUT, workspace_dir, used_default)


ACTIONS = {
    "run_file":     run_file,
    "run_code":     run_code,
    "save_and_run": save_and_run,
    "pip_install":  pip_install,
}


def dispatch(payload: dict, workspace_dir: str) -> dict:
    action = payload.get("action")
    if not action:
        return {"success": False, "error": "Missing 'action' field"}
    fn = ACTIONS.get(action)
    if fn is None:
        return {"success": False, "error": f"Unknown action: {action!r}. Valid: {list(ACTIONS)}"}
    ws = Path(workspace_dir)
    ws.mkdir(parents=True, exist_ok=True)
    kwargs = {k: v for k, v in payload.items() if k != "action"}
    try:
        result = fn(workspace_dir=ws, **kwargs)
        return {"success": True, "result": result}
    except TypeError as e:
        return {"success": False, "error": f"Bad arguments: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tools/code.py <workspace_dir> '<json payload>'")
        sys.exit(1)
    try:
        payload = json.loads(sys.argv[2])
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"Invalid JSON: {e}"}))
        sys.exit(1)
    print(json.dumps(dispatch(payload, sys.argv[1])))
