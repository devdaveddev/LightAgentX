"""
Example 6: Agent Snapshots — Portable Stateful Agents
=====================================================

Demonstrates how to export an agent with its full state (memory, config,
tool manifest) and import it into a different context — like Docker for agents.

No other lightweight agent framework offers this!
"""

from lightagentx import OpenAILLM, SingleAgent, AgentSnapshot, tool


# --- Define tools ---
@tool
def search(query: str) -> str:
    """Search for information.

    Args:
        query: The search query.
    """
    return f"Results for: {query}"


# --- Build and use an agent ---
llm = OpenAILLM()
agent = SingleAgent(
    name="ResearchBot",
    llm=llm,
    tools=[search],
    system_prompt="You are a research assistant with deep domain knowledge.",
    description="Expert researcher that finds and synthesizes information.",
)

print("--- Using agent on Project A ---")
result = agent.run("Find information about quantum computing applications")
print(f"Result: {result}\n")

# --- Export the agent (full state with memory) ---
agent.snapshot("research_bot.agent.json")
print("Agent saved to research_bot.agent.json")

# --- Export just the persona (no memory, for sharing) ---
AgentSnapshot.export_portable(agent, "research_bot_card.json")
print("Portable agent card saved to research_bot_card.json\n")

# --- On another project: restore the agent ---
print("--- Loading agent on Project B ---")
new_llm = OpenAILLM()  # can be a different instance/config
restored = SingleAgent.from_snapshot(
    "research_bot.agent.json",
    llm=new_llm,
    tools=[search],  # provide the same tools
)

# The agent remembers everything from Project A!
result = restored.run("Based on our previous research, what areas look most promising?")
print(f"Result: {result}")
