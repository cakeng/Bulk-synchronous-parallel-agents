from __future__ import annotations
import base64
import hashlib
import pickle as _pickle
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI


class AgentState(dict):
    """A plain dict with convenience helpers for use inside operator ``run()`` methods.

    Passed as ``_local`` to every operator.  Behaves exactly like a dict for all
    read/write/iteration operations, so existing code needs no changes.
    """

    def clear_history(self) -> None:
        """Clear the chat context and call log, keeping all other agent variables.

        Also clears ``opencode_session_id`` (and the legacy ``claude_session_id``)
        so that run_agent_opencode starts a fresh session rather than resuming
        the old one.
        """
        cfg = self.get("agent_config")
        if cfg is not None:
            cfg["context"] = []
            cfg["call_log"] = []
        self.pop("opencode_session_id", None)
        self.pop("claude_session_id", None)


def compute_unique_id(state: dict) -> str:
    """Compute a stable 8-char base64 ID for an agent state (sha256, last 8 chars).

    The key ``unique_id`` itself is excluded from the hash to avoid circularity.
    """
    state_for_hash = {k: v for k, v in state.items() if k != "unique_id"}
    try:
        raw = _pickle.dumps(state_for_hash, protocol=4)
    except Exception:
        raw = repr(sorted(state_for_hash.items())).encode()
    digest = hashlib.sha256(raw).digest()
    return base64.urlsafe_b64encode(digest).decode()[-8:]


# Default LLM server config (matches vllm_state.yaml)
DEFAULT_LLM_CONFIG: Dict[str, Any] = {
    "base_url": "http://127.0.0.1:18000/v1",
    "api_key": "12f34rtfq34er234T34GARW5G",
    "model": "Qwen/Qwen3.5-27B",
}


class Agent:
    """Single agent with an isolated variable dictionary.

    Variables are accessed and set via dict-style syntax:
        context = agent["agent_config"]["context"]
        agent["result"] = "some value"

    Or attribute-style (for convenience):
        context = agent.agent_config["context"]
        agent.result = "some value"

    Built-in keys always present:
        agent_rank  : int  — rank (position) assigned by the engine
        agent_config: dict — LLM connection info and chat context; built-in keys:
            base_url : str  — vLLM server URL
            api_key  : str  — API key
            model    : str  — model name
            context  : list — list of {"role": ..., "content": ...} dicts
    """

    def __init__(self, agent_rank: int, llm_state: Optional[Dict[str, Any]] = None):
        # Store everything in a flat dict; access via __getitem__/__setitem__
        d: Dict[str, Any] = {
            "agent_rank": agent_rank,
            "agent_config": {"context": [], **DEFAULT_LLM_CONFIG, **(llm_state or {})},
        }
        d["unique_id"] = compute_unique_id(d)
        object.__setattr__(self, "_vars", d)

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
                      Defaults to the agent's current context.
            **kwargs: Extra arguments forwarded to
                      AsyncOpenAI.chat.completions.create()
                      (e.g. temperature, max_tokens, stream, …).

        Returns:
            openai.types.chat.ChatCompletion response object.
        """
        cfg = self._vars["agent_config"]
        if messages is None:
            messages = cfg["context"]

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
        return f"Agent(rank={self._vars.get('agent_rank')}, vars={list(self._vars.keys())})"
