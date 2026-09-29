"""
Example 2: Tool-Calling Agent

Demonstrates:
  - The @tool decorator for creating tools
  - SingleAgent with tools
  - The tool-calling loop in action

Watch the colored console output to see how the agent:
  1. Receives your question
  2. Decides which tool to call
  3. Executes the tool
  4. Uses the result to form an answer
"""

from lightagentx import OpenAILLM, SingleAgent, tool


# ── Define tools using the @tool decorator ─────────────────────

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A math expression like '2 + 3 * 4' or 'pow(2, 10)'.
    """
    try:
        # Using eval with a limited namespace for safety
        allowed = {"__builtins__": {}, "pow": pow, "abs": abs, "round": round}
        result = eval(expression, allowed)
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: Name of the city to get weather for.
    """
    # Mock weather data (no API needed for demo)
    weather_data = {
        "london": "12°C, Cloudy with light rain",
        "new york": "18°C, Partly sunny",
        "tokyo": "22°C, Clear skies",
        "paris": "15°C, Overcast",
        "sydney": "25°C, Sunny and warm",
    }
    city_lower = city.lower()
    if city_lower in weather_data:
        return f"Weather in {city}: {weather_data[city_lower]}"
    return f"Weather data not available for {city}. Available cities: {', '.join(weather_data.keys())}"


@tool
def get_time(timezone: str) -> str:
    """Get the current time in a timezone.

    Args:
        timezone: Timezone name like 'UTC', 'EST', 'IST', 'JST'.
    """
    from datetime import datetime, timezone as tz, timedelta

    offsets = {
        "utc": 0, "est": -5, "cst": -6, "mst": -7, "pst": -8,
        "ist": 5.5, "jst": 9, "cet": 1, "gmt": 0, "aest": 11,
    }
    tz_lower = timezone.lower()
    offset = offsets.get(tz_lower)
    if offset is not None:
        hours = int(offset)
        minutes = int((offset - hours) * 60)
        now = datetime.now(tz.utc) + timedelta(hours=hours, minutes=minutes)
        return f"Current time in {timezone.upper()}: {now.strftime('%I:%M %p, %B %d, %Y')}"
    return f"Unknown timezone: {timezone}. Available: {', '.join(offsets.keys())}"


def main():
    # Create the LLM
    llm = OpenAILLM(model="gpt-4o-mini")

    # Create an agent with tools
    agent = SingleAgent(
        name="Assistant",
        llm=llm,
        tools=[calculate, get_weather, get_time],
        system_prompt=(
            "You are a helpful assistant with access to tools. "
            "Use tools when you need to calculate, check weather, or tell time. "
            "Always explain your reasoning."
        ),
        verbose=True,  # Show the colored reasoning trace
    )

    # Run some queries
    print("\n" + "=" * 60)
    print("🤖 LightAgentX Tool Agent Demo")
    print("=" * 60)

    queries = [
        "What's 25 * 37 + 12?",
        "What's the weather like in Tokyo and London?",
        "What time is it in IST and what's 2 to the power of 16?",
    ]

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"📝 Query: {query}")
        print("=" * 60)
        result = agent.run(query)
        print(f"\n📋 Final Answer: {result}")
        agent.reset()


if __name__ == "__main__":
    main()
