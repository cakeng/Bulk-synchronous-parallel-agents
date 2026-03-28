from abc import ABC, abstractmethod
from .agent import Agent


class Operator(ABC):
    """Abstract base class for all operators.

    An operator encapsulates a single step of computation that runs
    identically on every agent in the engine (data-parallel style).

    Subclass this and implement ``run``.  Inside ``run`` you can read
    agent variables with ``agent["var"]`` and write new ones with
    ``agent["var"] = value``.  Reading a variable that does not exist
    raises a ``KeyError`` immediately so you catch missing state early.

    Example operator file (operators/my_step.py):

        from src.operator import Operator

        class MyStep(Operator):
            async def run(self, agent):
                history = agent["chat_history"]
                history.append({"role": "user", "content": "Hello"})
                response = await agent.chat(messages=history)
                reply = response.choices[0].message.content
                history.append({"role": "assistant", "content": reply})
                agent["chat_history"] = history
    """

    @abstractmethod
    async def run(self, agent: "Agent") -> None:
        """Execute this operator on a single agent.

        Mutate ``agent`` in place — all changes are persisted to
        ``engine_state.pt`` after the step completes.
        """
