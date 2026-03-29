"""Example operator that does NOT call the LLM — useful for testing the framework
without a running vLLM server.

"""
from src.operator import Operator


class ExampleLocal(Operator):
    async def run(self, _local, _global):
        step = _global["step"]

        _local["agent_config"]["context"].append(
            {"role": "user", "content": f"This is engine step {step}."}
        )

        print(f"Agent {_local['agent_rank']} — engine step {step}.")
