"""
Example 7: Lifecycle Hooks — Event-Driven Middleware
====================================================

Demonstrates how to register callbacks for agent lifecycle events like
LLM calls, tool executions, errors, and agent boundaries — enabling
custom logging, guardrails, cost tracking, and more.

No other lightweight agent framework offers this!
"""

from lightagentx import OpenAILLM, SingleAgent, HookRegistry, tool


# --- Define tools ---
@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression.

    Args:
        expression: The math expression to evaluate.
    """
    return str(eval(expression))


# --- Create a hook registry with custom callbacks ---
hooks = HookRegistry()

# Track costs and prompts
call_log = []


@hooks.on("before_llm_call")
def log_prompt(event):
    msg_count = len(event.data["messages"])
    print(f"  [HOOK] Sending {msg_count} messages to {event.data['model']}")


@hooks.on("after_llm_call")
def log_response(event):
    response = event.data["response"]
    has_tools = "tool_calls" if response.has_tool_calls else "text"
    print(f"  [HOOK] LLM returned: {has_tools}")
    call_log.append({
        "model": event.data["model"],
        "type": has_tools,
    })


@hooks.on("before_tool_call")
def log_tool_start(event):
    print(f"  [HOOK] Calling tool: {event.data['tool_name']}({event.data['arguments']})")


@hooks.on("after_tool_call")
def log_tool_result(event):
    print(f"  [HOOK] Tool result: {event.data['result']}")


@hooks.on("on_agent_start")
def agent_start(event):
    print(f"\n  [HOOK] Agent '{event.data['agent_name']}' starting")


@hooks.on("on_agent_end")
def agent_end(event):
    print(f"  [HOOK] Agent '{event.data['agent_name']}' finished\n")


@hooks.on("on_error")
def on_error(event):
    print(f"  [HOOK] ERROR: {event.data['error']}")


# --- Create agent with hooks ---
llm = OpenAILLM()
agent = SingleAgent(
    name="MathBot",
    llm=llm,
    tools=[calculate],
    system_prompt="You are a math expert. Use the calculate tool for math.",
    hooks=hooks,
)

# --- Run the agent — hooks fire automatically ---
print("=== Running agent with lifecycle hooks ===")
result = agent.run("What is 42 * 58 + 17?")
print(f"Final answer: {result}")

# --- Review the call log ---
print(f"\n=== Call Log ({len(call_log)} LLM calls) ===")
for i, entry in enumerate(call_log, 1):
    print(f"  #{i}: model={entry['model']}, response_type={entry['type']}")
