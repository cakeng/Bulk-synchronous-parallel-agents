from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class Operator(ABC):
    """Default operator — modifies agent state, no return value.

    async def run(self, _local: dict, _global: dict) -> None
    """

    OPERATOR_TYPE: str = "base"

    @abstractmethod
    async def run(self, _local: dict, _global: dict) -> None:
        """Mutate ``_local`` and/or ``_global`` in place.
        Must not return a value (implicit ``return None``).
        """


class ForkOperator(Operator):
    """Replicates each agent N times, where N is the int returned by ``run``.

    After all agents finish the engine deep-copies each agent's state N times.
    Returning 0 removes the agent (0 copies produced).

    Post-fork additions to each child's _local:
        parent_id : unique_id of the agent that was forked
        fork_rank : 0-based rank among the siblings from the same parent

    async def run(self, _local: dict, _global: dict) -> int  (non-negative)
    """

    OPERATOR_TYPE: str = "fork"

    @abstractmethod
    async def run(self, _local: dict, _global: dict) -> int:
        """Return the number of copies to create (0 = remove this agent)."""


class KillOperator(Operator):
    """Removes agents that return ``True``.

    The engine raises ``RuntimeError`` if every agent returns ``True``
    (refusing to kill all agents).

    async def run(self, _local: dict, _global: dict) -> bool
    """

    OPERATOR_TYPE: str = "kill"

    @abstractmethod
    async def run(self, _local: dict, _global: dict) -> bool:
        """Return ``True`` to remove this agent, ``False`` to keep it."""


class SortOperator(Operator):
    """Reorders agents by the float each one returns (descending).

    Agent IDs are reassigned sequentially after reordering.

    async def run(self, _local: dict, _global: dict) -> float
    """

    OPERATOR_TYPE: str = "sort"

    @abstractmethod
    async def run(self, _local: dict, _global: dict) -> float:
        """Return a score; agents are ordered highest-score-first."""


class ShuffleOperator(Operator):
    """Lets each agent gather outputs from a specified set of other agents.

    Each agent returns ``(obj, [rank, ...])`` where ``obj`` is the value this
    agent wants to share, and the list contains the ``agent_rank``s of the
    agents whose shared objects this agent wants to receive.

    After all agents finish the engine populates each agent's _local with:
        shuffle_output : dict[agent_rank -> deep_copy(obj)]

    Use ``_global["agent_size"]`` to collect from all agents:
        return (my_obj, list(range(_global["agent_size"])))

    async def run(self, _local: dict, _global: dict) -> tuple[Any, list[int]]
    """

    OPERATOR_TYPE: str = "shuffle"

    @abstractmethod
    async def run(self, _local: dict, _global: dict) -> Tuple[Any, List[int]]:
        """Return ``(my_obj, [rank, ...])``."""
