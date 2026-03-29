"""Code generation for BSA operators.

The UI editor shows only the *body* of ``async def run`` — no class or
function wrapper.  This module handles the round-trip:

* ``generate_full_code(op_name, body, op_type)``  — body ➜ full .py file
* ``extract_body(full_code)``                     — full .py file ➜ body
* ``detect_op_type(full_code)``                   — reads the BSA:OP_TYPE tag

Auto-unpack / auto-pack
-----------------------
Every plain variable name that is *read* in the body is unpacked from
``_local`` before execution::

    my_var = _local.get('my_var')

Every plain variable name that is *written* in the body (plus all that were
unpacked) is packed back into ``_local`` inside a ``finally`` block, so the
pack still runs even when the body returns early::

    try:
        <body>
    finally:
        try: _local['my_var'] = my_var
        except NameError: pass

Names starting with ``_`` (e.g. ``_tmp``) are excluded from auto-management.
To read an engine global, write ``_global["step"]`` or the shorthand
``_global.step``; the latter is rewritten to the former during code
generation and is excluded from ``_local`` auto-unpack.
"""
from __future__ import annotations

import ast
import re
import textwrap

# ── Sentinel tags embedded in generated files ─────────────────────────────────
_TAG_OP_TYPE   = "# <<BSA:OP_TYPE={}>>"
_TAG_BODY_START = "# <<BSA:BODY_START>>"
_TAG_BODY_END   = "# <<BSA:BODY_END>>"

# ── Operator type → base class mapping ───────────────────────────────────────
_BASE_CLASS: dict[str, str] = {
    "base":    "Operator",
    "fork":    "ForkOperator",
    "kill":    "KillOperator",
    "sort":    "SortOperator",
    "shuffle": "ShuffleOperator",
}

# ── Default editor bodies for each operator type ─────────────────────────────
DEFAULT_BODIES: dict[str, str] = {
    "base": """\
_parsed, _raw, _thinking, _tool_calls, _tokens = await run_agent(
    user_input=f"Hello!",
    output_config={"reply": str},
    agent_state=_local,
)
last_reply = _parsed["reply"]
""",
    "fork": """\
# Return the number of child copies to create
import random

num = random.randint(1, 5)
_parsed, _raw, _thinking, _tool_calls, _tokens = await run_agent(
    user_input=f"Give me {num} different cat species.",
    output_config={"answer": list[str]},
    agent_state=_local,
)
cats = _parsed["answer"]
return len(cats)
""",
    "kill": """\
# Return True to remove this agent
import random

def char_to_int(char: str) -> int:
    return ord(char) - ord('A') + 1

num = random.randint(1, 10)
_parsed, _raw, _thinking, _tool_calls, _tokens = await run_agent(
    user_input=f"What is the {num}th largest state in the United States?",
    output_config={"answer": str},
    agent_state=_local,
)
state = _parsed["answer"]
state_int = char_to_int(state[0])
return state_int < 10
""",
    "sort": """\
# Return a float score; agents are ordered highest-first
import random

def char_to_int(char: str) -> int:
    return ord(char) - ord('A') + 1

num = random.randint(1, 10)
_parsed, _raw, _thinking, _tool_calls, _tokens = await run_agent(
    user_input=f"Who is the {num}th president of the United States?",
    output_config={"answer": str},
    agent_state=_local,
)
president = _parsed["answer"]
president_int = char_to_int(president[0])
return float(president_int)
""",
    "shuffle": """\
# Return (my_object, [ranks_to_collect_from])
import random

num = random.randint(1, 5)
_parsed, _raw, _thinking, _tool_calls, _tokens = await run_agent(
    user_input=f"Give me {num} different sharks.",
    output_config={"answer": list[str]},
    agent_state=_local,
)
sharks = _parsed["answer"]
return (sharks, list(range(_global.agent_size)))
""",
}

# ── Names never auto-unpacked or auto-packed ──────────────────────────────────
_SKIP: frozenset[str] = frozenset({
    "_local", "_global", "self",
    "True", "False", "None",
    "print", "len", "range", "list", "dict", "set", "tuple", "frozenset",
    "str", "int", "float", "bool", "bytes", "bytearray", "type",
    "isinstance", "issubclass", "hasattr", "getattr", "setattr", "delattr",
    "zip", "enumerate", "map", "filter", "sorted", "reversed",
    "max", "min", "sum", "abs", "round", "divmod", "pow", "hash",
    "any", "all", "next", "iter", "open", "input", "repr",
    "super", "object", "property", "classmethod", "staticmethod",
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "RuntimeError", "StopIteration", "NotImplementedError",
    "OSError", "IOError", "FileNotFoundError", "PermissionError",
    "asyncio", "run_agent",
    "Operator", "ForkOperator", "KillOperator", "SortOperator", "ShuffleOperator",
})


# ── Public API ────────────────────────────────────────────────────────────────

def generate_full_code(op_name: str, body: str, op_type: str = "base") -> str:
    """Wrap *body* in a complete operator .py file with auto-unpack/pack."""
    base_class = _BASE_CLASS.get(op_type, "Operator")
    class_name = _make_class_name(op_name)

    # Rewrite _global.attr (non-call) → _global["attr"]
    normalised = _rewrite_global_attrs(body)

    read_vars, written_vars = _analyse(normalised)
    pack_vars = read_vars | written_vars   # pack everything that was touched

    ind = "        "   # 8-space indent (inside class + method)

    unpack_lines = [f"{ind}{v} = _local.get('{v}')" for v in sorted(read_vars)]
    # Each pack line is wrapped in try/except so missing names (e.g. after an
    # early return that skips some assignments) don't crash the finally block.
    pack_lines = [
        f"{ind}    try: _local['{v}'] = {v}\n{ind}    except NameError: pass"
        for v in sorted(pack_vars)
    ]

    body_indented = textwrap.indent(
        textwrap.dedent(normalised).strip(),
        f"{ind}    ",   # 12 spaces — inside try:
    )

    lines: list[str] = [
        f"from src.operator import {base_class}",
        "from src.run_agent import run_agent",
        "",
        _TAG_OP_TYPE.format(op_type),
        "",
        f"class {class_name}({base_class}):",
        f"    async def run(self, _local, _global):",
    ]

    if unpack_lines:
        lines.append(f"{ind}# auto-unpacked from agent state")
        lines.extend(unpack_lines)
        lines.append("")

    lines.append(f"{ind}try:")
    lines.append(f"{ind}    {_TAG_BODY_START}")
    lines.append(body_indented)
    lines.append(f"{ind}    {_TAG_BODY_END}")
    lines.append(f"{ind}finally:")
    if pack_lines:
        lines.append(f"{ind}    # auto-packed back to agent state")
        lines.extend(pack_lines)
    else:
        lines.append(f"{ind}    pass")

    lines.append("")
    return "\n".join(lines)


def extract_body(full_code: str) -> str:
    """Return the user-authored body from a generated operator file."""
    src_lines = full_code.splitlines()
    try:
        start = next(i for i, l in enumerate(src_lines) if _TAG_BODY_START in l)
        end   = next(i for i, l in enumerate(src_lines) if _TAG_BODY_END   in l)
        raw   = src_lines[start + 1 : end]
    except StopIteration:
        raw = _fallback_extract_run_body(src_lines)
    return textwrap.dedent("\n".join(raw)).strip()


def detect_op_type(full_code: str) -> str:
    """Read the BSA:OP_TYPE tag from a generated file, defaulting to 'base'."""
    for line in full_code.splitlines():
        m = re.search(r"<<BSA:OP_TYPE=(\w+)>>", line)
        if m:
            return m.group(1)
    return "base"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _make_class_name(op_name: str) -> str:
    stem = re.sub(r"\.py$", "", op_name)
    words = re.split(r"[_\-\s]+", stem)
    return "".join(w.capitalize() for w in words) or "Op"


def _rewrite_global_attrs(body: str) -> str:
    """Replace ``_global.attr`` (not a method call) with ``_global["attr"]``.

    The word boundary ``\\b`` after the group prevents the backtracking that
    would otherwise let the engine match a truncated prefix like ``ge`` from
    ``get(`` when the full token is rejected by the negative lookahead.
    """
    return re.sub(r'\b_global\.([A-Za-z_]\w*)\b(?!\s*\()', r'_global["\1"]', body)


def _analyse(body: str) -> tuple[set[str], set[str]]:
    """Return ``(read_vars, written_vars)`` for auto-managed plain Name nodes.

    Excludes:
    * names in ``_SKIP``
    * names starting with ``_``
    * names introduced by ``import`` / ``from … import`` in the body
    """
    try:
        tree = ast.parse(textwrap.dedent(body))
    except SyntaxError:
        return set(), set()

    # Collect imported names so we don't shadow them with _local lookups
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    imported.add(alias.asname or alias.name)

    skip = _SKIP | imported

    read_vars:    set[str] = set()
    written_vars: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Name):
            continue
        name = node.id
        if name in skip or name.startswith("_"):
            continue
        if isinstance(node.ctx, ast.Store):
            written_vars.add(name)
        elif isinstance(node.ctx, ast.Load):
            read_vars.add(name)

    return read_vars, written_vars


def _fallback_extract_run_body(lines: list[str]) -> list[str]:
    """Heuristic fallback: pull the body of ``async def run(...)`` from source."""
    in_run   = False
    body:    list[str] = []
    run_ind: int | None = None

    for line in lines:
        stripped = line.lstrip()
        if not in_run:
            if re.match(r"async\s+def\s+run\s*\(", stripped):
                in_run = True
            continue
        if stripped and run_ind is None:
            run_ind = len(line) - len(stripped)
        if run_ind is not None and stripped and (len(line) - len(stripped)) < run_ind:
            break
        body.append(line)

    return body
