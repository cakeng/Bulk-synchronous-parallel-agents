#!/usr/bin/env python3
"""Bash tool — run shell commands in the agent workspace."""

TOOL_NAME = "bash"

TOOL_PROMPTS = {
    "full": """\
--- Tool: bash ---
Execute bash shell commands. The working directory is your workspace directory.

  run   Run a shell command.
    {"tool_call": {"tool": "bash", "action": "run", "command": "<shell command>", "timeout": 30}}

Returns: stdout, stderr, exit_code, timed_out.
timeout is in seconds (default 30). Increase for long-running commands.
Use the filesystem tool for simple file operations; use bash for pipelines,
package managers, compilers, or anything that needs a full shell environment.
""",
}

TOOL_SCHEMAS = {
    "full": [
        {
            "type": "function",
            "function": {
                "name": "bash__run",
                "description": (
                    "Execute a bash command with the workspace directory as the working directory. "
                    "Returns stdout, stderr, exit_code, and timed_out."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Shell command to execute",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (default 30). Increase for long-running commands.",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
    ],
}

import json
import subprocess
import sys
from pathlib import Path


def run(command: str, workspace_dir: Path, timeout: int = 30) -> dict:
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "exit_code": -1,
            "timed_out": True,
        }


ACTIONS = {"run": run}


def dispatch(payload: dict, workspace_dir: str) -> dict:
    action = payload.get("action")
    if action not in ACTIONS:
        return {"success": False, "error": f"Unknown action: {action!r}. Available: {sorted(ACTIONS)}"}
    ws = Path(workspace_dir)
    ws.mkdir(parents=True, exist_ok=True)
    kwargs = {k: v for k, v in payload.items() if k != "action"}
    try:
        result = ACTIONS[action](workspace_dir=ws, **kwargs)
        return {"success": True, "result": result}
    except TypeError as e:
        return {"success": False, "error": f"Bad parameters for {action!r}: {e}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tools/bash.py <workspace_dir> '<json payload>'")
        sys.exit(1)
    try:
        payload = json.loads(sys.argv[2])
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"Invalid JSON: {e}"}))
        sys.exit(1)
    print(json.dumps(dispatch(payload, sys.argv[1])))
