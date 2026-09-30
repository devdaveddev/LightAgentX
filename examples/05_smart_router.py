"""
Example 5: SmartRouter — Auto Model Switching
==============================================

Demonstrates how SmartRouter automatically selects the best LLM from a pool
based on task complexity, round-robin cycling, or fallback strategies.

No other lightweight agent framework offers this!
"""

from lightagentx import OpenAILLM, SmartRouter, SingleAgent


# --- Setup: Pool of models (cheapest first) ---
router = SmartRouter(
    models=[
        OpenAILLM(model="gpt-4o-mini"),    # fast + cheap
        OpenAILLM(model="gpt-4o"),          # powerful + expensive
    ],
    strategy="complexity",  # auto-detect task complexity
)

agent = SingleAgent(
    name="SmartBot",
    llm=router,
    system_prompt="You are a helpful assistant.",
)

# --- Simple question → routes to gpt-4o-mini ---
print("--- Simple Question ---")
result = agent.run("What is 2 + 2?")
print(f"Answer: {result}\n")

# --- Complex question → routes to gpt-4o ---
print("--- Complex Question ---")
result = agent.run(
    "Analyze the trade-offs between microservices and monolithic architecture. "
    "Compare them step by step across scalability, maintainability, deployment "
    "complexity, and team coordination. Provide a comprehensive evaluation."
)
print(f"Answer: {result}\n")

# --- Check which model handled each request ---
print("--- Routing History ---")
for entry in router.routing_history:
    print(f"  Call #{entry['call_number']}: {entry['model']} "
          f"(strategy={entry['strategy']}, took {entry['elapsed_seconds']}s)")
