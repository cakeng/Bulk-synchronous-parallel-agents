"""KillOperator example: remove agents whose score is below a threshold.

The engine refuses to run if every agent would be killed.

"""
from src.operator import KillOperator
from src.run_agent import run_agent
import random


class ExampleKill(KillOperator):
    def char_to_int(self, char: str) -> int:
        return ord(char) - ord('A') + 1

    async def run(self, _local, _global):

        parsed, raw, thinking, tool_calls, tokens = await run_agent(
            user_input    = f"What is the {_local['agent_rank'] + 1}th largest state in the United States?",
            output_config = {"answer": str, "explanation": str},
            agent_state   = _local,
        )
        state = parsed["answer"]
        state_int = self.char_to_int(state[0])
        _local["state"] = (state, state_int)
        return float(state_int) < 10
