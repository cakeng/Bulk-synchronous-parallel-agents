"""Tool registry — dynamically loads tools/ modules and provides a unified interface.

Usage from an operator:
    from src import tools

    # Dispatch a tool call using the agent's workspace
    result = tools.dispatch("filesystem", {"action": "read_file", "path": "output.txt"}, workspace_dir)

    # Dispatch from a native function call (tool_name__action → tool + action)
    result = tools.dispatch_function_call("filesystem__read_file", {"path": "output.txt"}, workspace_dir)

    # Build system prompt and schemas (done automatically on agent init)
    prompt  = tools.build_system_prompt(workspace_dir)
    schemas = tools.build_tool_schemas()
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"

# tool_name -> loaded module
_modules: dict[str, Any] = {}


def _load_tools() -> None:
    """Scan tools/ and import every non-private .py module once."""
    if _modules:
        return
    for path in sorted(_TOOLS_DIR.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"_bsa_tools.{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
        except Exception as exc:
            # Don't hard-fail if a single tool has an import problem
            import warnings
            warnings.warn(f"Could not load tool '{path.name}': {exc}", stacklevel=2)
            continue
        name = getattr(mod, "TOOL_NAME", path.stem)
        _modules[name] = mod


def build_system_prompt(workspace_dir: str) -> str:
    """Build the full system prompt for an agent, including the workspace path and all tool docs."""
    _load_tools()
    parts = [
        "You are an agent in a multi-agent framework.",
        f"Your workspace directory is:\n  {workspace_dir}",
        "All file paths in tool calls are relative to this directory. "
        "The directory already exists — store all files and outputs there.\n",
    ]
    for name in sorted(_modules):
        mod = _modules[name]
        prompts = getattr(mod, "TOOL_PROMPTS", {})
        if "full" in prompts:
            parts.append(prompts["full"])
    return "\n".join(parts)


def build_tool_schemas() -> list:
    """Return the combined list of OpenAI-compatible tool schemas for all tools."""
    _load_tools()
    schemas: list = []
    for name in sorted(_modules):
        mod = _modules[name]
        tool_schemas = getattr(mod, "TOOL_SCHEMAS", {})
        if "full" in tool_schemas:
            schemas.extend(tool_schemas["full"])
    return schemas


def dispatch(tool_name: str, payload: dict, workspace_dir: str) -> dict:
    """Route a tool call to the appropriate tool module.

    payload must contain an 'action' key identifying the specific action.
    workspace_dir is the absolute path to the agent's workspace directory.
    """
    _load_tools()
    mod = _modules.get(tool_name)
    if mod is None:
        return {"success": False, "error": f"Unknown tool: {tool_name!r}. Available: {sorted(_modules)}"}
    dispatch_fn = getattr(mod, "dispatch", None)
    if dispatch_fn is None:
        return {"success": False, "error": f"Tool {tool_name!r} has no dispatch() function"}
    return dispatch_fn(payload, workspace_dir)


def dispatch_function_call(function_name: str, arguments: dict, workspace_dir: str) -> dict:
    """Dispatch a native OpenAI function call (tool__action naming convention).

    Example:
        dispatch_function_call("filesystem__read_file", {"path": "out.txt"}, workspace_dir)
    """
    if "__" not in function_name:
        return {"success": False, "error": f"Invalid function name (expected 'tool__action'): {function_name!r}"}
    tool_name, action = function_name.split("__", 1)
    payload = {"action": action, **arguments}
    return dispatch(tool_name, payload, workspace_dir)
