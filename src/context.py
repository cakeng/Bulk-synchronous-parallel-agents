"""Context: lightweight wrapper around a list of chat messages.

Usage inside an operator::

    from src.context import Context

    ctx = Context(_local["agent_config"]["context"])
    ctx.user("Hello, what is 2 + 2?")
    ...
    _local["agent_config"]["context"] = ctx.messages
"""
from __future__ import annotations

from typing import List, Dict


class Context:
    """Wraps a list of chat messages with helpers for appending turns.

    Args:
        messages: Existing list of ``{"role": ..., "content": ...}`` dicts.
                  A shallow copy is made so the original list is not mutated.

    Attributes:
        messages: The working message list — updated by ``run_agent`` and
                  written back to ``_local["agent_config"]["context"]`` on success.
    """

    def __init__(self, messages: List[Dict[str, str]]) -> None:
        self.messages: List[Dict[str, str]] = list(messages)

    def user(self, content: str) -> "Context":
        """Append a user message and return self for optional chaining."""
        self.messages.append({"role": "user", "content": content})
        return self

    def assistant(self, content: str) -> "Context":
        """Append an assistant message and return self for optional chaining."""
        self.messages.append({"role": "assistant", "content": content})
        return self

    def system(self, content: str) -> "Context":
        """Append a system message and return self for optional chaining."""
        self.messages.append({"role": "system", "content": content})
        return self
