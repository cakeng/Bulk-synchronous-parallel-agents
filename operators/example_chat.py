"""Example operator: sends a single user message and records the assistant reply.

Run:
    python step_engine.py operators/example_chat.py
"""
from src.operator import Operator


class ExampleChat(Operator):
    async def run(self, agent):
        # Read the agent's chat history (built-in variable, always present)
        history = agent["chat_history"]

        # Append the user turn
        history.append({"role": "user", "content": "What is 2 + 2? Reply in one sentence."})

        # Call the LLM (uses agent's llm_config by default)
        response = await agent.chat(messages=history)
        reply = response.choices[0].message.content

        # Append the assistant turn
        history.append({"role": "assistant", "content": reply})

        # Write back — this persists to the next step
        agent["chat_history"] = history
        agent["last_reply"] = reply

        print(f"Agent {agent['agent_id']} — LLM replied: {reply}")
