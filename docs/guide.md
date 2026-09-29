# LightAgentX — Usage Guide

A complete guide to using every component of the framework.

---

## Table of Contents

- [Installation](#installation)
- [1. LLM — Talk to Any Provider](#1-llm--talk-to-any-provider)
- [2. Memory — Conversation History](#2-memory--conversation-history)
- [3. Tools — Give Your Agent Abilities](#3-tools--give-your-agent-abilities)
- [4. Agent Loop — The Core Engine](#4-agent-loop--the-core-engine)
- [5. SingleAgent — The Building Block](#5-singleagent--the-building-block)
- [6. SequentialPipeline — Chain Agents](#6-sequentialpipeline--chain-agents)
- [7. CrewAgent — Hierarchical Delegation](#7-crewagent--hierarchical-delegation)
- [8. ReAct Planner — Reasoning + Acting](#8-react-planner--reasoning--acting)
- [9. Logger — Visual Tracing](#9-logger--visual-tracing)
- [API Reference](#api-reference)

---

## Installation

```bash
# From PyPI
pip install lightagentx

# From source (development)
git clone https://github.com/devdaveddev/LightAgentX.git
cd lightagentx
pip install -e ".[dev]"
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

---

## 1. LLM — Talk to Any Provider

### Basic Chat

```python
from lightagentx import OpenAILLM

llm = OpenAILLM()  # defaults: model="gpt-4o-mini", temperature=0.7

# Simple text completion
response = llm.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
])
print(response.content)
```

### Configure the Model

```python
llm = OpenAILLM(
    model="gpt-4o",          # any OpenAI model
    temperature=0.2,          # lower = more deterministic
    max_tokens=2048,          # max response length
    api_key="sk-...",         # or use OPENAI_API_KEY env var
)
```

### Use a Custom LLM Provider

Subclass `BaseLLM` to add any provider (Anthropic, Gemini, local models):

```python
from lightagentx import BaseLLM, LLMResponse

class MyCustomLLM(BaseLLM):
    def __init__(self):
        super().__init__(model="my-model")

    def chat(self, messages):
        # Call your API here
        text = my_api_call(messages)
        return LLMResponse(content=text)

    def chat_with_tools(self, messages, tools):
        # Call your API with tool schemas
        result = my_api_call_with_tools(messages, tools)
        return LLMResponse(
            content=result.text,
            tool_calls=[
                {"id": "call_1", "name": "tool_name", "arguments": {"key": "value"}}
            ],
        )
```

### LLMResponse Object

Every LLM call returns an `LLMResponse`:

```python
response = llm.chat(messages)

response.content        # str — the text response
response.tool_calls     # list[dict] — tool calls requested by the LLM
response.has_tool_calls # bool — True if tool_calls is non-empty
response.raw            # Any — raw provider response (for debugging)
```

---

## 2. Memory — Conversation History

### BufferMemory (Sliding Window)

Keeps the last N messages. Simple, fast, predictable.

```python
from lightagentx import BufferMemory

memory = BufferMemory(max_messages=20)

# System messages are stored separately and NEVER evicted
memory.add_message("system", "You are a helpful assistant.")

# Regular messages
memory.add_message("user", "Hello!")
memory.add_message("assistant", "Hi there!")

# Get all messages (system + recent)
messages = memory.get_messages()
# → [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]

# Clear everything
memory.clear()
```

### SummaryMemory (LLM-Compressed)

When messages exceed the limit, older ones are summarized by the LLM instead of being dropped.

```python
from lightagentx import SummaryMemory, OpenAILLM

llm = OpenAILLM()
memory = SummaryMemory(llm=llm, max_messages=10)

memory.add_message("system", "You are a research assistant.")

# Add many messages... when > 10, older messages get auto-summarized
for i in range(15):
    memory.add_message("user", f"Research point {i}")
    memory.add_message("assistant", f"Analysis of point {i}")

# The summary is injected into the system prompt automatically
messages = memory.get_messages()

# Access the running summary directly
print(memory.summary)
```

**When to use which:**

| Memory | Use When | Tradeoff |
|---|---|---|
| `BufferMemory` | Short conversations, cost-sensitive | Fast, but loses old context |
| `SummaryMemory` | Long conversations, context matters | Preserves semantics, costs extra LLM calls |

---

## 3. Tools — Give Your Agent Abilities

### Define a Tool

Use the `@tool` decorator. It auto-extracts the name, description, and parameter schema from your function:

```python
from lightagentx import tool

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A math expression like '2 + 3 * 4'.
    """
    return str(eval(expression))

@tool
def get_weather(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city.

    Args:
        city: The city name.
        units: Temperature units (celsius/fahrenheit).
    """
    return f"{city}: 22°C, Clear"
```

The decorator generates OpenAI-compatible JSON Schema automatically:
- **Name** ← `func.__name__`
- **Description** ← first line of docstring
- **Parameters** ← type hints + `Args:` docstring section
- **Required/Optional** ← based on whether params have default values

### Inspect a Tool

```python
print(calculate.name)          # "calculate"
print(calculate.description)   # "Evaluate a mathematical expression."
print(calculate.parameters)    # {"type": "object", "properties": {...}, "required": [...]}

# View the full OpenAI schema
import json
print(json.dumps(calculate.to_openai_schema(), indent=2))
```

### Call a Tool Directly

```python
result = calculate(expression="2 + 3")
print(result)  # "5"
```

### Supported Types

| Python Type | JSON Schema Type |
|---|---|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |
| `list` | `"array"` |
| `dict` | `"object"` |

### ToolRegistry — Manage Multiple Tools

```python
from lightagentx import ToolRegistry

registry = ToolRegistry()
registry.register(calculate)
registry.register(get_weather)

# Or register many at once
registry.register_many([calculate, get_weather])

# Lookup
tool = registry.get("calculate")

# Get all schemas for the OpenAI API
schemas = registry.to_openai_schema()

# List names
print(registry.tool_names)  # ["calculate", "get_weather"]
```

### ToolExecutor — Run Tool Calls

```python
from lightagentx import ToolExecutor

executor = ToolExecutor(registry)

# Execute a single tool call (as returned by the LLM)
result = executor.execute({
    "id": "call_123",
    "name": "calculate",
    "arguments": {"expression": "25 * 37"},
})
print(result)  # "925"

# Execute multiple tool calls
results = executor.execute_many([
    {"id": "c1", "name": "calculate", "arguments": {"expression": "2+2"}},
    {"id": "c2", "name": "get_weather", "arguments": {"city": "Tokyo"}},
])
# → [{"tool_call_id": "c1", "content": "4"}, {"tool_call_id": "c2", "content": "Tokyo: 22°C, Clear"}]
```

> **Error handling:** If a tool raises an exception, the executor returns an error string instead of crashing. This lets the LLM recover.

---

## 4. Agent Loop — The Core Engine

The `AgentLoop` is the heart of the framework. It implements the universal agent cycle:

```
LLM → check for tool calls → execute tools → feed results back → repeat
```

### Basic Usage

```python
from lightagentx import AgentLoop, OpenAILLM, tool

llm = OpenAILLM()

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

loop = AgentLoop(
    llm=llm,
    tools=[multiply],
    system_prompt="You are a math assistant.",
    max_iterations=10,   # safety limit
    verbose=True,        # colored output
)

answer = loop.run("What is 25 * 37?")
print(answer)  # "25 × 37 = 925"
```

### How It Works

```
1. User input → added to memory
2. LOOP:
   ├─ Get messages from memory
   ├─ Send to LLM (with tool schemas)
   ├─ If LLM returns tool_calls:
   │   ├─ Execute each tool
   │   ├─ Add results to memory
   │   └─ CONTINUE LOOP
   └─ If LLM returns text:
       └─ Return as final answer
3. Safety: stops after max_iterations
```

### Reset

```python
loop.reset()  # clears memory, start fresh
```

---

## 5. SingleAgent — The Building Block

A `SingleAgent` wraps an `AgentLoop` with a name and persona. It's the primary way to create agents.

```python
from lightagentx import SingleAgent, OpenAILLM, tool

llm = OpenAILLM()

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

agent = SingleAgent(
    name="ResearchBot",
    llm=llm,
    tools=[search_web],
    system_prompt="You are an expert researcher. Always cite your sources.",
    description="Expert at finding and analyzing information",  # used by multi-agent systems
    max_iterations=10,
    verbose=True,
)

result = agent.run("What are the latest advances in quantum computing?")
print(result)

# Reset for a fresh conversation
agent.reset()
```

---

## 6. SequentialPipeline — Chain Agents

Pass output from one agent to the next, like a production line.

```
Input → Agent A → output → Agent B → output → Agent C → Final Output
```

```python
from lightagentx import SingleAgent, SequentialPipeline, OpenAILLM

llm = OpenAILLM()

researcher = SingleAgent(
    name="Researcher",
    llm=llm,
    description="Finds and summarizes information",
    system_prompt="You research topics and provide detailed findings.",
)

writer = SingleAgent(
    name="Writer",
    llm=llm,
    description="Writes polished articles from research",
    system_prompt="You write engaging articles based on research provided to you.",
)

editor = SingleAgent(
    name="Editor",
    llm=llm,
    description="Edits and improves articles",
    system_prompt="You edit articles for clarity, grammar, and style.",
)

pipeline = SequentialPipeline(
    name="Content Pipeline",
    agents=[researcher, writer, editor],
)

article = pipeline.run("Write about the future of AI agents")
print(article)
```

---

## 7. CrewAgent — Hierarchical Delegation

A manager LLM analyzes the task, delegates to specialist agents, and synthesizes results.

```python
from lightagentx import SingleAgent, CrewAgent, OpenAILLM

llm = OpenAILLM()

tech_expert = SingleAgent(
    name="TechExpert",
    llm=llm,
    description="Expert at explaining technical concepts",
    system_prompt="You explain complex tech topics clearly.",
)

business_analyst = SingleAgent(
    name="BusinessAnalyst",
    llm=llm,
    description="Expert at analyzing business impact",
    system_prompt="You analyze business implications and market trends.",
)

crew = CrewAgent(
    name="Analysis Team",
    agents=[tech_expert, business_analyst],
    manager_llm=llm,  # LLM that coordinates the team
)

report = crew.run("Analyze the impact of large language models on software development")
print(report)
```

**How it works:**
1. **Manager** receives agent descriptions and creates a JSON delegation plan
2. **Specialists** each run their assigned subtask independently
3. **Manager** synthesizes all results into a unified answer

**Fallback:** If the manager outputs invalid JSON, the task is automatically broadcast to all agents.

---

## 8. ReAct Planner — Reasoning + Acting

The ReAct pattern forces the LLM to think before acting:

```
Thought: I need to look up the population → Action: search → Observation: 1.4 billion → repeat...
```

```python
from lightagentx import ReActPlanner, OpenAILLM

llm = OpenAILLM()
planner = ReActPlanner(llm)

step = planner.plan(
    goal="Find the population of India",
    context="",
    available_tools=["search_web", "calculate"],
)

print(step.thought)       # "I need to search for India's population"
print(step.action)        # "search_web"
print(step.action_input)  # "population of India 2024"
```

The planner returns a `Step` object:

```python
from lightagentx import Step

step = Step(
    thought="reasoning text",
    action="tool_name",        # or "Final Answer"
    action_input="tool args",  # or the final answer text
)
```

---

## 9. Logger — Visual Tracing

The colored logger shows exactly what the agent is doing:

```python
from lightagentx import AgentLogger

logger = AgentLogger(verbose=True)

logger.system("Agent initialized")       # ⚙️ [SYSTEM] cyan
logger.thought("I should search first")  # 💭 [THOUGHT] yellow
logger.action("search", {"q": "AI"})     # 🔧 [ACTION] magenta
logger.observation("Found 10 results")   # 👁️ [OBSERVATION] blue
logger.result("Here is the answer...")   # ✅ [RESULT] green
logger.error("Something went wrong")     # ❌ [ERROR] red
logger.agent("Bot1", "Starting task")    # 🤖 [AGENT] cyan
logger.plan("Delegation plan", "...")    # 📋 [PLAN] yellow
logger.separator()                        # ──────────
```

Set `verbose=False` on any agent or loop to silence all output.

---

## API Reference

### Core Classes

| Class | Import | Description |
|---|---|---|
| `OpenAILLM` | `from lightagentx import OpenAILLM` | OpenAI ChatCompletion provider |
| `BaseLLM` | `from lightagentx import BaseLLM` | Abstract base — subclass for custom providers |
| `LLMResponse` | `from lightagentx import LLMResponse` | Unified return type from all LLMs |
| `BufferMemory` | `from lightagentx import BufferMemory` | Sliding window conversation memory |
| `SummaryMemory` | `from lightagentx import SummaryMemory` | LLM-compressed conversation memory |
| `BaseTool` | `from lightagentx import BaseTool` | Tool data class |
| `@tool` | `from lightagentx import tool` | Decorator to create tools from functions |
| `ToolRegistry` | `from lightagentx import ToolRegistry` | Name-based tool lookup |
| `ToolExecutor` | `from lightagentx import ToolExecutor` | Dispatches LLM tool calls to Python |
| `AgentLoop` | `from lightagentx import AgentLoop` | Core reasoning loop |
| `SingleAgent` | `from lightagentx import SingleAgent` | One LLM + tools + persona |
| `SequentialPipeline` | `from lightagentx import SequentialPipeline` | Chain agents A → B → C |
| `CrewAgent` | `from lightagentx import CrewAgent` | Manager delegates to specialists |
| `ReActPlanner` | `from lightagentx import ReActPlanner` | Thought → Action → Observation planner |
| `Step` | `from lightagentx import Step` | Single reasoning step |
| `AgentLogger` | `from lightagentx import AgentLogger` | Colored console logger |

### Key Methods

| Method | On | Description |
|---|---|---|
| `.run(input)` | `SingleAgent`, `SequentialPipeline`, `CrewAgent`, `AgentLoop` | Run with input, return output string |
| `.reset()` | `SingleAgent`, `AgentLoop` | Clear memory and start fresh |
| `.chat(messages)` | `BaseLLM` subclasses | Text completion |
| `.chat_with_tools(messages, tools)` | `BaseLLM` subclasses | Completion with function-calling |
| `.add_message(role, content)` | `BufferMemory`, `SummaryMemory` | Add a message to history |
| `.get_messages()` | `BufferMemory`, `SummaryMemory` | Get conversation history |
| `.clear()` | `BufferMemory`, `SummaryMemory` | Clear all messages |
| `.plan(goal, context, tools)` | `ReActPlanner` | Get next reasoning step |
| `.register(tool)` | `ToolRegistry` | Register a tool |
| `.execute(tool_call)` | `ToolExecutor` | Execute a single tool call |
