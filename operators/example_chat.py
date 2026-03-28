"""Example operator: LLM call with structured output enforcement via run_agent().

Asks the model a simple arithmetic question and enforces a JSON response with
two fields: ``answer`` (int) and ``explanation`` (str).

"""
from src.operator import Operator
from src.run_agent import run_agent


class ExampleChat(Operator):
    async def run(self, _local, _global):
        step = _global["step"]

        parsed, raw, thinking, tool_calls, tokens = await run_agent(
            user_input    = f"What is {_local['agent_rank']} + {step}?",
            output_config = {"answer": int, "explanation": str},
            agent_config  = _local["llm_state"],
        )

        _local["rank_step_answer"] = parsed["answer"]
