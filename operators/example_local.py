"""Example operator that does NOT call the LLM — useful for testing the framework
without a running vLLM server.

Run:
    python step_engine.py operators/example_local.py
"""
from src.operator import Operator


class ExampleLocal(Operator):
    async def run(self, agent):
        # Increment a step counter (works even on first run via .get())
        agent["step"] = agent.get("step", 0) + 1

        # Append a dummy message to chat_history
        history = agent["chat_history"]
        history.append({"role": "user", "content": f"This is step {agent['step']}."})
        agent["chat_history"] = history

        print(f"Agent {agent['agent_id']} completed local step {agent['step']}.")
