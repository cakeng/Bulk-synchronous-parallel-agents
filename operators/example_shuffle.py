"""ShuffleOperator example: each agent generates a list of mammals and shares
it with every other agent.

After this step every agent has a ``shuffle_output`` dict:
    _local["shuffle_output"] == {rank: [shark, ...], ...}

"""
from src.operator import ShuffleOperator
from src.run_agent import run_agent
import random


class ExampleShuffle(ShuffleOperator):

    async def run(self, _local, _global):
        parsed, raw, thinking, tool_calls, tokens = await run_agent(
            user_input    = f"Give me a list of {_local['agent_rank'] + 1} different sharks.",
            output_config = {"answer": list[str], "explanation": str},
            agent_state   = _local,
        )

        _local["sharks"] = parsed["answer"]

        # Share my shark list and collect from every agent by rank
        all_ranks = list(range(_global["agent_size"]))
        return (_local["sharks"], all_ranks)
