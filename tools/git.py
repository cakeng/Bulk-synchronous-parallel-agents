#!/usr/bin/env python3
"""
Git tool — run git commands inside directories within the agent workspace.

Usage:
  python tools/git.py <workspace_dir> '{"action": "run", "repo": "my_project", "command": "status"}'
"""

TOOL_NAME = "git"

TOOL_PROMPTS = {
    "full": """\
--- Tool: git ---
Run git commands inside repository directories within your workspace.
All repo paths are relative to your workspace directory.

  run   Run a git command in the specified repo directory.
    {"tool_call": {"tool": "git", "action": "run", "repo": "<subdir>", "command": "<git subcommand and args>"}}

Examples:
  {"tool_call": {"tool": "git", "action": "run", "repo": "my_project", "command": "init"}}
  {"tool_call": {"tool": "git", "action": "run", "repo": "my_project", "command": "add -A"}}
  {"tool_call": {"tool": "git", "action": "run", "repo": "my_project", "command": "commit -m 'Initial commit'"}}
  {"tool_call": {"tool": "git", "action": "run", "repo": "my_project", "command": "log --oneline -10"}}

Notes:
  - repo is relative to your workspace directory. Use "." to target the workspace root itself.
  - command is everything after "git" (e.g. "status", "add -A", "commit -m 'msg'").
  - stdout and stderr are both returned along with the exit code.
  - git user.name and user.email are set automatically.
""",
}

TOOL_SCHEMAS = {
    "full": [
        {
            "type": "function",
            "function": {
                "name": "git__run",
                "description": "Run a git command in a repository directory within the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "repo": {
                            "type": "string",
                            "description": "Repo path relative to workspace (e.g. 'my_project' or '.')",
                        },
                        "command": {
                            "type": "string",
                            "description": "Git subcommand and args — everything after 'git' (e.g. 'status', 'add -A')",
                        },
                    },
                    "required": ["repo", "command"],
                },
            },
        },
    ],
}

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

_BLOCKED_SUBCOMMANDS = {"daemon", "fast-import", "http-backend", "shell"}


def _safe_repo_path(repo: str, workspace_dir: Path) -> Path:
    raw = repo.strip().lstrip("/")
    p = (workspace_dir / raw).resolve() if raw and raw != "." else workspace_dir.resolve()
    if not str(p).startswith(str(workspace_dir.resolve())):
        raise ValueError(f"Repo path outside workspace: {repo!r}")
    return p


def run_git(repo: str, command: str, workspace_dir: Path) -> dict:
    repo_path = _safe_repo_path(repo, workspace_dir)
    repo_path.mkdir(parents=True, exist_ok=True)

    try:
        args = shlex.split(command)
    except ValueError as e:
        return {"success": False, "error": f"Could not parse command: {e}"}

    if not args:
        return {"success": False, "error": "command must not be empty"}

    subcommand = args[0].lower().lstrip("-")
    if subcommand in _BLOCKED_SUBCOMMANDS:
        return {"success": False, "error": f"git subcommand {args[0]!r} is not allowed"}

    env = {
        **os.environ,
        "GIT_AUTHOR_NAME":    "Agent",
        "GIT_AUTHOR_EMAIL":   "agent@localhost",
        "GIT_COMMITTER_NAME": "Agent",
        "GIT_COMMITTER_EMAIL":"agent@localhost",
    }

    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        return {
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "repo": str(repo_path.relative_to(workspace_dir.resolve())),
            "command": "git " + command,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "git command timed out after 60s"}
    except FileNotFoundError:
        return {"success": False, "error": "git executable not found on PATH"}


ACTIONS = {"run": run_git}


def dispatch(payload: dict, workspace_dir: str) -> dict:
    action = payload.get("action")
    if action not in ACTIONS:
        return {"success": False, "error": f"Unknown action: {action!r}. Available: {sorted(ACTIONS)}"}
    ws = Path(workspace_dir)
    ws.mkdir(parents=True, exist_ok=True)
    kwargs = {k: v for k, v in payload.items() if k != "action"}
    try:
        return ACTIONS[action](workspace_dir=ws, **kwargs)
    except TypeError as e:
        return {"success": False, "error": f"Bad parameters for {action!r}: {e}"}
    except (ValueError, PermissionError) as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tools/git.py <workspace_dir> '<json payload>'")
        sys.exit(1)
    try:
        payload = json.loads(sys.argv[2])
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"Invalid JSON: {e}"}))
        sys.exit(1)
    print(json.dumps(dispatch(payload, sys.argv[1])))
