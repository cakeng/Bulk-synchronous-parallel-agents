"""SortOperator example: reorder agents by score (descending).

After this step agent IDs are reassigned so that agent 0 has the highest score.

"""
from src.operator import SortOperator
from src.run_agent import run_agent
import random


class ExampleSort(SortOperator):

    def char_to_int(self, char: str) -> int:
        return ord(char) - ord('A') + 1

    async def run(self, _local, _global):

        parsed, raw, thinking, tool_calls, tokens = await run_agent(
            user_input    = f"Who is the {_local['agent_rank'] + 1}th president of the United States?",
            output_config = {"answer": str, "explanation": str},
            agent_config  = _local["llm_state"],
        )
        president = parsed["answer"]
        president_int = self.char_to_int(president[0])
        _local["president"] = (president, president_int)
        return float(president_int)
