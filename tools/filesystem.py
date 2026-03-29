#!/usr/bin/env python3
"""
Filesystem tool — read/write/search within the agent workspace.

Usage:
  python tools/filesystem.py <workspace_dir> '{"action": "list_directory", "path": "."}'
"""

TOOL_NAME = "filesystem"

TOOL_PROMPTS = {
    "full": """\
--- Tool: filesystem ---
Read and write files in your workspace. All paths are relative to your workspace directory.

  list_directory  List contents of a directory.
    {"tool_call": {"tool": "filesystem", "action": "list_directory", "path": "<rel_path>", "recursive": false}}

  read_file       Read a file (max 500 KB). Content is prefixed with line numbers for use with replace_lines.
    {"tool_call": {"tool": "filesystem", "action": "read_file", "path": "<rel_path>"}}

  read_lines      Read a specific line range (1-based, inclusive).
    {"tool_call": {"tool": "filesystem", "action": "read_lines", "path": "<rel_path>", "start": 10, "end": 50}}

  write_file      Write (create or overwrite) a file.
    {"tool_call": {"tool": "filesystem", "action": "write_file", "path": "<rel_path>", "content": "<content>"}}

  replace_lines   Replace a line range in an existing file (1-based, inclusive).
    {"tool_call": {"tool": "filesystem", "action": "replace_lines", "path": "<rel_path>", "start": 10, "end": 20, "content": "<replacement>"}}

  search_files    Glob-match filenames under a directory.
    {"tool_call": {"tool": "filesystem", "action": "search_files", "pattern": "<glob>", "directory": "<rel_path>"}}

  grep_files      Regex-search file contents.
    {"tool_call": {"tool": "filesystem", "action": "grep_files", "pattern": "<regex>", "directory": "<rel_path>", "file_glob": "*.txt"}}

  append_lines_from_file  Copy lines from one file into another at a given position.
                          dest_line=0 appends at end. Optionally restrict source with src_start/src_end.
    {"tool_call": {"tool": "filesystem", "action": "append_lines_from_file", "src_path": "<rel>", "dest_path": "<rel>", "dest_line": 10}}
""",
}

TOOL_SCHEMAS = {
    "full": [
        {
            "type": "function",
            "function": {
                "name": "filesystem__list_directory",
                "description": "List contents of a directory relative to the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative path to directory"},
                        "recursive": {"type": "boolean", "description": "List recursively (default false)"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "filesystem__read_file",
                "description": "Read a file's contents (max 500 KB). Path relative to workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative path to file"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "filesystem__read_lines",
                "description": "Read a specific line range of a file (1-based, inclusive).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative path to file"},
                        "start": {"type": "integer", "description": "First line to read (1-based)"},
                        "end": {"type": "integer", "description": "Last line to read (1-based, inclusive)"},
                    },
                    "required": ["path", "start", "end"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "filesystem__write_file",
                "description": "Write (create or overwrite) a file in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative path to file"},
                        "content": {"type": "string", "description": "Content to write"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "filesystem__replace_lines",
                "description": "Replace a line range in an existing file (1-based, inclusive).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Relative path to file"},
                        "start": {"type": "integer", "description": "First line to replace (1-based)"},
                        "end": {"type": "integer", "description": "Last line to replace (1-based, inclusive)"},
                        "content": {"type": "string", "description": "Replacement text"},
                    },
                    "required": ["path", "start", "end", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "filesystem__search_files",
                "description": "Glob-match filenames under a directory in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Glob pattern"},
                        "directory": {"type": "string", "description": "Directory to search in"},
                    },
                    "required": ["pattern", "directory"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "filesystem__grep_files",
                "description": "Regex-search file contents under a directory in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regex pattern to search for"},
                        "directory": {"type": "string", "description": "Directory to search in"},
                        "file_glob": {"type": "string", "description": "File glob filter (e.g. *.txt)"},
                    },
                    "required": ["pattern", "directory"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "filesystem__append_lines_from_file",
                "description": (
                    "Copy lines from a source file into a destination file. "
                    "Lines are inserted before dest_line (1-based); dest_line=0 appends at end."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "src_path": {"type": "string", "description": "Source file (relative to workspace)"},
                        "dest_path": {"type": "string", "description": "Destination file (relative to workspace)"},
                        "dest_line": {"type": "integer", "description": "Insert before this line (1-based). 0 = append at end."},
                        "src_start": {"type": "integer", "description": "First line to copy from src (1-based, default 1)"},
                        "src_end": {"type": "integer", "description": "Last line to copy from src (1-based, inclusive, default = last line)"},
                    },
                    "required": ["src_path", "dest_path", "dest_line"],
                },
            },
        },
    ],
}

import json
import re
import sys
from pathlib import Path

MAX_FILE_SIZE = 500 * 1024  # 500 KB


# ---------------------------------------------------------------------------
# Path helpers (all scoped to workspace_dir)
# ---------------------------------------------------------------------------

def _safe_path(raw: str, workspace_dir: Path) -> Path:
    """Resolve path confined to workspace_dir."""
    stripped = raw.lstrip("/")
    p = (workspace_dir / stripped).resolve() if stripped else workspace_dir.resolve()
    if not str(p).startswith(str(workspace_dir.resolve())):
        raise ValueError(f"Path traversal denied: {raw!r}")
    return p


def _rel(p: Path, workspace_dir: Path) -> str:
    """Return path relative to workspace_dir for display."""
    try:
        return str(p.relative_to(workspace_dir.resolve()))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def list_directory(path: str, workspace_dir: Path, recursive: bool = False) -> dict:
    p = _safe_path(path, workspace_dir)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    def _tree(d: Path) -> dict:
        result = {}
        try:
            entries = sorted(d.iterdir())
        except PermissionError:
            return {"__error__": "permission denied"}
        for entry in entries:
            if recursive and entry.is_dir():
                result[entry.name + "/"] = _tree(entry)
            elif entry.is_dir():
                result[entry.name + "/"] = {}
            else:
                result[entry.name] = entry.stat().st_size
        return result

    return {"path": _rel(p, workspace_dir), "tree": _tree(p)}


def _number_lines(lines: list, start: int = 1) -> str:
    width = len(str(start + len(lines) - 1))
    parts = []
    for i, line in enumerate(lines, start):
        parts.append(f"{i:{width}}  {line}" if line.endswith("\n") else f"{i:{width}}  {line}\n")
    return "".join(parts)


def read_file(path: str, workspace_dir: Path) -> dict:
    p = _safe_path(path, workspace_dir)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not p.is_file():
        raise IsADirectoryError(f"Not a file: {path}")
    size = p.stat().st_size
    if size > MAX_FILE_SIZE:
        raise ValueError(f"File too large ({size} bytes > {MAX_FILE_SIZE} limit)")
    lines = p.read_text(errors="replace").splitlines(keepends=True)
    return {"path": _rel(p, workspace_dir), "content": _number_lines(lines), "total_lines": len(lines), "size": size}


def read_lines(path: str, workspace_dir: Path, start: int, end: int) -> dict:
    p = _safe_path(path, workspace_dir)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not p.is_file():
        raise IsADirectoryError(f"Not a file: {path}")
    if start < 1:
        raise ValueError(f"start must be >= 1, got {start}")
    if end < start:
        raise ValueError(f"end ({end}) must be >= start ({start})")
    lines = p.read_text(errors="replace").splitlines(keepends=True)
    total = len(lines)
    selected = lines[start - 1:end]
    return {
        "path": _rel(p, workspace_dir),
        "start": start,
        "end": min(end, total),
        "total_lines": total,
        "content": _number_lines(selected, start),
    }


def write_file(path: str, workspace_dir: Path, content: str) -> dict:
    p = _safe_path(path, workspace_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return {"path": _rel(p, workspace_dir), "bytes_written": len(content.encode())}


def replace_lines(path: str, workspace_dir: Path, start: int, end: int, content: str) -> dict:
    p = _safe_path(path, workspace_dir)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if start < 1:
        raise ValueError(f"start must be >= 1, got {start}")
    if end < start:
        raise ValueError(f"end ({end}) must be >= start ({start})")
    lines = p.read_text(errors="replace").splitlines(keepends=True)
    total_before = len(lines)
    replacement_lines = content.splitlines(keepends=True)
    if replacement_lines and not replacement_lines[-1].endswith("\n") and end < total_before:
        replacement_lines[-1] += "\n"
    new_lines = lines[:start - 1] + replacement_lines + lines[end:]
    new_content = "".join(new_lines)
    p.write_text(new_content)
    return {
        "path": _rel(p, workspace_dir),
        "replaced_lines": f"{start}-{end}",
        "total_lines_before": total_before,
        "total_lines_after": len(new_lines),
        "bytes_written": len(new_content.encode()),
    }


def append_lines_from_file(
    src_path: str,
    dest_path: str,
    workspace_dir: Path,
    dest_line: int,
    src_start: int = 1,
    src_end: int | None = None,
) -> dict:
    src  = _safe_path(src_path,  workspace_dir)
    dest = _safe_path(dest_path, workspace_dir)
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    if src_start < 1:
        raise ValueError(f"src_start must be >= 1, got {src_start}")
    src_lines = src.read_text(errors="replace").splitlines(keepends=True)
    end_idx = src_end if src_end is not None else len(src_lines)
    if end_idx < src_start:
        raise ValueError(f"src_end ({end_idx}) must be >= src_start ({src_start})")
    inserted = src_lines[src_start - 1:end_idx]
    if inserted and not inserted[-1].endswith("\n"):
        inserted[-1] += "\n"
    dest_lines = dest.read_text(errors="replace").splitlines(keepends=True) if dest.exists() else []
    total_before = len(dest_lines)
    if dest_line == 0 or dest_line > total_before:
        new_lines = dest_lines + inserted
    else:
        new_lines = dest_lines[:dest_line - 1] + inserted + dest_lines[dest_line - 1:]
    dest.parent.mkdir(parents=True, exist_ok=True)
    new_content = "".join(new_lines)
    dest.write_text(new_content)
    return {
        "src_path": _rel(src, workspace_dir),
        "dest_path": _rel(dest, workspace_dir),
        "lines_inserted": len(inserted),
        "total_lines_before": total_before,
        "total_lines_after": len(new_lines),
        "bytes_written": len(new_content.encode()),
    }


def search_files(pattern: str, workspace_dir: Path, directory: str = ".") -> dict:
    base = _safe_path(directory, workspace_dir)
    if not base.exists():
        return {"pattern": pattern, "directory": directory, "matches": [], "note": "directory does not exist"}
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    matches = sorted(_rel(p, workspace_dir) for p in base.rglob(pattern))
    return {"pattern": pattern, "directory": directory, "matches": matches}


def grep_files(pattern: str, workspace_dir: Path, directory: str = ".", file_glob: str = "*") -> dict:
    base = _safe_path(directory, workspace_dir)
    if not base.exists():
        return {"pattern": pattern, "directory": directory, "matches": {}, "note": "directory does not exist"}
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")
    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise ValueError(f"Invalid regex: {e}")
    results = {}
    for filepath in sorted(base.rglob(file_glob)):
        if not filepath.is_file() or filepath.stat().st_size > MAX_FILE_SIZE:
            continue
        try:
            lines = filepath.read_text(errors="replace").splitlines()
        except Exception:
            continue
        hits = [f"{i+1}: {line}" for i, line in enumerate(lines) if rx.search(line)]
        if hits:
            results[_rel(filepath, workspace_dir)] = hits
    return {"pattern": pattern, "directory": directory, "matches": results}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

ACTIONS = {
    "list_directory": list_directory,
    "read_file": read_file,
    "read_lines": read_lines,
    "write_file": write_file,
    "replace_lines": replace_lines,
    "append_lines_from_file": append_lines_from_file,
    "search_files": search_files,
    "grep_files": grep_files,
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
        print("Usage: python tools/filesystem.py <workspace_dir> '<json payload>'")
        sys.exit(1)
    try:
        payload = json.loads(sys.argv[2])
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"Invalid JSON: {e}"}))
        sys.exit(1)
    print(json.dumps(dispatch(payload, sys.argv[1])))
