from __future__ import annotations
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI


# Default LLM server config (matches vllm_config.yaml)
DEFAULT_LLM_CONFIG: Dict[str, Any] = {
    "base_url": "http://127.0.0.1:18000/v1",
    "api_key": "12f34rtfq34er234T34GARW5G",
    "model": "Qwen/Qwen3.5-27B",
}


class Agent:
    """Single agent with an isolated variable dictionary.

    Variables are accessed and set via dict-style syntax:
        history = agent["chat_history"]
        agent["result"] = "some value"

    Or attribute-style (for convenience):
        history = agent.chat_history
        agent.result = "some value"

    Built-in keys always present:
        agent_id    : int  — unique id assigned by the engine
        llm_config  : dict — connection info for the LLM server
        chat_history: list — list of {"role": ..., "content": ...} dicts
    """

    def __init__(self, agent_id: int, llm_config: Optional[Dict[str, Any]] = None):
        # Store everything in a flat dict; access via __getitem__/__setitem__
        object.__setattr__(self, "_vars", {
            "agent_id": agent_id,
            "llm_config": {**DEFAULT_LLM_CONFIG, **(llm_config or {})},
            "chat_history": [],
        })

    # ------------------------------------------------------------------
    # Dict-style access
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        try:
            return self._vars[key]
        except KeyError:
            available = list(self._vars.keys())
            raise KeyError(
                f"Agent variable '{key}' not found. "
                f"Available variables: {available}"
            ) from None

    def __setitem__(self, key: str, value: Any) -> None:
        self._vars[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._vars

    def get(self, key: str, default: Any = None) -> Any:
        return self._vars.get(key, default)

    # ------------------------------------------------------------------
    # Attribute-style access (delegates to _vars for non-private names)
    # ------------------------------------------------------------------

    def __getattr__(self, key: str) -> Any:
        # Only called when normal attribute lookup fails
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self._vars[key]
        except KeyError:
            available = list(self._vars.keys())
            raise AttributeError(
                f"Agent has no variable '{key}'. "
                f"Available variables: {available}"
            ) from None

    def __setattr__(self, key: str, value: Any) -> None:
        if key.startswith("_"):
            object.__setattr__(self, key, value)
        else:
            self._vars[key] = value

    # ------------------------------------------------------------------
    # State serialization (used by Engine / worker)
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Return a copy of the internal variable dict for serialization."""
        return dict(self._vars)

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore the internal variable dict from a deserialized state."""
        object.__getattribute__(self, "_vars").clear()
        object.__getattribute__(self, "_vars").update(state)

    # ------------------------------------------------------------------
    # LLM interface (OpenAI-compatible, async)
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ):
        """Call the LLM server with an OpenAI chat completions request.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
                      Defaults to the agent's current chat_history.
            **kwargs: Extra arguments forwarded to
                      AsyncOpenAI.chat.completions.create()
                      (e.g. temperature, max_tokens, stream, …).

        Returns:
            openai.types.chat.ChatCompletion response object.
        """
        cfg = self._vars["llm_config"]
        if messages is None:
            messages = self._vars["chat_history"]

        client = AsyncOpenAI(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
        )
        model = kwargs.pop("model", cfg["model"])
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )
        return response

    def __repr__(self) -> str:
        return f"Agent(id={self._vars.get('agent_id')}, vars={list(self._vars.keys())})"
