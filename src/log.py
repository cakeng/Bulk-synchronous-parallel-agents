"""ANSI color helpers for BSA framework terminal output."""
import pprint
import sys

# ANSI codes
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_CYAN   = "\033[36m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_BLUE   = "\033[94m"   # bright blue
_MAGENTA= "\033[35m"
_RED    = "\033[31m"
_DIM    = "\033[2m"

def _c(code: str, text: str) -> str:
    return f"{code}{text}{_RESET}"

def engine(msg: str) -> str:   return _c(_BOLD + _CYAN,    msg)
def agent_out(msg: str) -> str:return _c(_GREEN,            msg)
def agent_err(msg: str) -> str:return _c(_YELLOW,           msg)
def verbose_in(msg: str) -> str: return _c(_BLUE,           msg)
def verbose_out(msg: str) -> str:return _c(_MAGENTA,        msg)
def error(msg: str) -> str:    return _c(_BOLD + _RED,      msg)
def debug(msg: str) -> str:    return _c(_BOLD + _YELLOW,   msg)
def dim(msg: str) -> str:      return _c(_DIM,              msg)
def bold(msg: str) -> str:     return _c(_BOLD,             msg)

def print_engine(msg: str) -> None:
    print(engine(msg))

def print_agent_out(prefix: str, line: str) -> None:
    print(agent_out(bold(prefix)) + " " + line)

def print_agent_err(prefix: str, line: str) -> None:
    print(agent_err(bold(prefix) + f" {line}"), file=sys.stderr)

def print_error(msg: str) -> None:
    print(error(msg), file=sys.stderr)

def print_debug(msg: str) -> None:
    print(debug(msg))

def format_value(v, full: bool = False) -> str:
    """Human-readable representation of a variable value.

    full=False (level 1): compact summary — list[N], {key, …}, short repr.
    full=True  (level 2): pretty-printed via pprint, multi-line for large objects.
    """
    if full:
        return pprint.pformat(v, width=72, depth=4)
    # compact
    if isinstance(v, list):
        return f"list[{len(v)}]"
    if isinstance(v, dict):
        keys = list(v.keys())
        preview = ", ".join(str(k) for k in keys[:4])
        suffix = ", …" if len(keys) > 4 else ""
        return f"{{{preview}{suffix}}}"
    s = repr(v)
    return s if len(s) <= 80 else s[:77] + "…"


def _print_block_lines(color_fn, prefix: str, key: str, value_str: str) -> None:
    """Print a key: value entry, indenting continuation lines of multi-line values."""
    lines = value_str.splitlines()
    print(color_fn(f"  │  {bold(key)}: {lines[0]}"))
    for extra in lines[1:]:
        print(color_fn(f"  │    {extra}"))


def print_agent_input(agent_id: int, state: dict, full: bool = False) -> None:
    """Print the agent's variable dict before operator execution (blue)."""
    tag = bold(f"[Agent {agent_id}]")
    print(verbose_in(f"  ┌─ {tag} INPUT STATE ─────────────────────────────"))
    for k, v in state.items():
        _print_block_lines(verbose_in, "  │  ", k, format_value(v, full=full))
    print(verbose_in(f"  └─────────────────────────────────────────────────"))


def print_agent_output_diff(
    agent_id: int, before: dict, after: dict, full: bool = False
) -> None:
    """Print new/changed variables after operator execution (magenta)."""
    tag = bold(f"[Agent {agent_id}]")
    new_keys     = [k for k in after if k not in before]
    changed_keys = [k for k in after if k in before and after[k] != before[k]]

    if not new_keys and not changed_keys:
        print(verbose_out(f"  ── {tag} OUTPUT: no variable changes ──────────────"))
        return

    print(verbose_out(f"  ┌─ {tag} OUTPUT CHANGES ──────────────────────────"))
    for k in changed_keys:
        before_str = format_value(before[k], full=full)
        after_str  = format_value(after[k],  full=full)
        if "\n" in before_str or "\n" in after_str:
            # Multi-line: show before and after on separate indented blocks
            print(verbose_out(f"  │  ~ {bold(k)} (before):"))
            for line in before_str.splitlines():
                print(verbose_out(f"  │      {line}"))
            print(verbose_out(f"  │    (after):"))
            for line in after_str.splitlines():
                print(verbose_out(f"  │      {line}"))
        else:
            print(verbose_out(f"  │  ~ {bold(k)}: {before_str}  →  {after_str}"))
    for k in new_keys:
        _print_block_lines(verbose_out, "  │  ", f"+ {k}", format_value(after[k], full=full))
    print(verbose_out(f"  └─────────────────────────────────────────────────"))
