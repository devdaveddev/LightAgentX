"""
Example 3: ReAct Agent

Demonstrates:
  - The ReAct (Reasoning + Acting) pattern
  - How the planner structures LLM reasoning into steps
  - Step-by-step problem solving with tools

The ReAct pattern forces the LLM to think before acting:
  Thought → Action → Observation → Thought → Action → ... → Final Answer
"""

from lightagentx import OpenAILLM, ReActPlanner, AgentLogger, tool
from lightagentx.tools import ToolRegistry, ToolExecutor


@tool
def search(query: str) -> str:
    """Search for information about a topic.

    Args:
        query: The search query.
    """
    # Mock search results
    data = {
        "python creator": "Python was created by Guido van Rossum in 1991.",
        "rust creator": "Rust was created by Graydon Hoare at Mozilla, first released in 2010.",
        "javascript creator": "JavaScript was created by Brendan Eich in 1995 at Netscape.",
        "guido van rossum": "Guido van Rossum is a Dutch programmer, creator of Python. Born January 31, 1956.",
        "brendan eich": "Brendan Eich is an American technologist. He co-founded Mozilla and created JavaScript.",
    }
    query_lower = query.lower()
    for key, value in data.items():
        if key in query_lower:
            return value
    return f"No results found for: {query}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A math expression to evaluate.
    """
    try:
        allowed = {"__builtins__": {}, "pow": pow, "abs": abs}
        return str(eval(expression, allowed))
    except Exception as e:
        return f"Error: {e}"


def main():
    llm = OpenAILLM(model="gpt-4o-mini", temperature=0.0)  # low temp for structured output
    logger = AgentLogger(verbose=True)

    # Set up planner and tools
    planner = ReActPlanner(llm)
    registry = ToolRegistry()
    registry.register_many([search, calculate])
    executor = ToolExecutor(registry, logger)

    # The goal
    goal = "Who created Python and JavaScript? What's the sum of their birth years?"

    print("=" * 60)
    print("ReAct Agent Demo")
    print(f"Goal: {goal}")
    print("=" * 60)

    context = ""
    max_steps = 8

    for step_num in range(1, max_steps + 1):
        logger.separator()
        logger.system(f"ReAct Step {step_num}/{max_steps}")

        # Plan the next step
        step = planner.plan(
            goal=goal,
            context=context,
            available_tools=registry.tool_names,
        )

        logger.thought(step.thought)

        # Check if we have a final answer
        if step.action.lower() == "final answer":
            logger.result(step.action_input)
            print(f"\nFinal Answer: {step.action_input}")
            return

        # Execute the action
        tool_call = {
            "id": f"step_{step_num}",
            "name": step.action,
            "arguments": _parse_action_input(step.action, step.action_input),
        }

        result = executor.execute(tool_call)

        # Build context for next step
        context += (
            f"\nStep {step_num}:\n"
            f"  Thought: {step.thought}\n"
            f"  Action: {step.action}\n"
            f"  Action Input: {step.action_input}\n"
            f"  Observation: {result}\n"
        )

    logger.error("Max steps reached without a final answer")


def _parse_action_input(action_name: str, action_input: str) -> dict:
    """Map the free-text action input to the tool's expected arguments."""
    # For our tools, the first parameter is the input
    tool_params = {
        "search": "query",
        "calculate": "expression",
    }
    param_name = tool_params.get(action_name, "input")
    return {param_name: action_input}


if __name__ == "__main__":
    main()
