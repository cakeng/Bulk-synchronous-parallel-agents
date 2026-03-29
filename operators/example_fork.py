"""ForkOperator example: expand each agent into N copies.

The number of forks equals the number of cats the model lists when asked.

"""
from src.operator import ForkOperator
from src.run_agent import run_agent
import random


class ExampleFork(ForkOperator):
    async def run(self, _local, _global):
        rand_int = random.randint(1, 4)

        parsed, raw, thinking, tool_calls, tokens = await run_agent(
            user_input    = f"Give me a list of {rand_int} different cats.",
            output_config = {"answer": list[str], "explanation": str},
            agent_state   = _local,
        )

        _local["cats"] = parsed["answer"]
        return len(_local["cats"])  # number of child copies to create
