# LightAgentX — Deep Dive Interview Guide

> A from-scratch agentic AI framework. Built to understand exactly how LangChain, CrewAI, and AutoGPT work under the hood. Every line is intentional.

---

## What is LightAgentX?

LightAgentX is a minimal but complete agentic AI framework written in pure Python. It has no magic, no hidden abstractions. Every concept that LangChain or CrewAI uses — tool calling, memory, planning, multi-agent orchestration — is implemented here from scratch with full visibility into how it works.

The only external dependency is `openai>=1.0.0`. Everything else is standard Python.

---

## Tech Stack

| Layer | What | Why |
|---|---|---|
| Language | Python 3.10+ | dataclasses, `from __future__ import annotations`, `get_type_hints` |
| LLM Provider | OpenAI Chat Completions API | Function calling / tool_calls support |
| SDK | `openai` Python SDK v1+ | `client.chat.completions.create()` |
| Type System | Python `typing` + `inspect` | Auto JSON Schema generation from type hints |
| No frameworks | No LangChain, no Pydantic | Everything hand-rolled |

---

## Folder Structure and What Each One Does

```
lightagentx/
├── llm/          # Abstraction over LLM providers + SmartRouter
├── memory/       # Conversation history storage
├── tools/        # @tool decorator, registry, executor
├── planning/     # ReAct reasoning pattern
├── loop/         # The core agent execution loop
├── agents/       # SingleAgent, SequentialPipeline, CrewAgent
├── hooks.py      # Lifecycle hooks — event-driven middleware
├── snapshot.py   # Agent snapshots — portable stateful agents
└── utils/        # Colored terminal logger
```

Each folder is a self-contained module with a `base.py` (abstract contract) and one or more concrete implementations.

---

## Module 1: `llm/` — The LLM Abstraction Layer

### Why it exists

You want to swap between OpenAI, Anthropic, Gemini without rewriting your agent. So you define a contract first, then implement it.

### `llm/base.py`

```python
@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: Any = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0
```

`LLMResponse` is the unified response object every LLM provider must return. It normalizes the difference between providers. `content` is the text reply. `tool_calls` is a list of dicts like `{"id": "call_abc", "name": "calculate", "arguments": {"expression": "2+2"}}`. `raw` stores the original SDK response for debugging.

```python
class BaseLLM(ABC):
    def __init__(self, model: str, temperature: float = 0.7, max_tokens: int = 1024): ...

    @abstractmethod
    def chat(self, messages: list[dict]) -> LLMResponse: ...

    @abstractmethod
    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> LLMResponse: ...
```

`BaseLLM` is the abstract contract. Any provider must implement `chat()` for plain text and `chat_with_tools()` for function calling. The `messages` format follows OpenAI's convention: `[{"role": "user", "content": "Hello"}]`.

### `llm/openai_llm.py`

```python
class OpenAILLM(BaseLLM):
    def __init__(self, model="gpt-4o-mini", temperature=0.7, max_tokens=1024, api_key=None):
        super().__init__(model, temperature, max_tokens)
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = OpenAI(api_key=resolved_key)
```

API key resolution: explicit arg wins, falls back to env var. Raises `ValueError` if neither exists.

```python
    def chat(self, messages):
        response = self._client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=self.temperature, max_tokens=self.max_tokens,
        )
        return LLMResponse(content=response.choices[0].message.content or "")
```

Plain chat: just calls the API and wraps the text in `LLMResponse`.

```python
    def chat_with_tools(self, messages, tools):
        kwargs = {"model": self.model, "messages": messages, ...}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0].message

        parsed_tool_calls = []
        if choice.tool_calls:
            for tc in choice.tool_calls:
                parsed_tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                })

        return LLMResponse(content=choice.content or "", tool_calls=parsed_tool_calls)
```

The key thing here: OpenAI returns `tool_calls[i].function.arguments` as a **JSON string**, not a dict. So `json.loads()` is required. This is a common gotcha. The result is a clean list of dicts that the rest of the framework uses.

---

## Module 2: `memory/` — Conversation History

### Why it exists

The OpenAI API is stateless. Every call needs the full conversation history. Memory modules manage that list of messages.

### `memory/base.py`

```python
class BaseMemory(ABC):
    @abstractmethod
    def add_message(self, role: str, content: str, **kwargs) -> None: ...

    @abstractmethod
    def get_messages(self) -> list[dict]: ...

    @abstractmethod
    def clear(self) -> None: ...

    def add_tool_message(self, tool_call_id: str, content: str) -> None:
        self.add_message("tool", content, tool_call_id=tool_call_id)

    def add_assistant_tool_calls(self, content: str, tool_calls: list[dict]) -> None:
        self.add_message("assistant", content, tool_calls=tool_calls)
```

The two convenience methods at the bottom are critical. When the LLM calls a tool, the API requires a very specific message sequence:
1. An `assistant` message with a `tool_calls` field
2. A `tool` message with a `tool_call_id` field matching the call

If you get this order wrong or miss a field, the API throws an error.

### `memory/buffer.py` — Sliding Window Memory

```python
class BufferMemory(BaseMemory):
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self._system_message: dict | None = None
        self._messages: list[dict] = []
```

System message is stored separately so it is **never evicted**. The sliding window only applies to user/assistant/tool messages.

```python
    def add_message(self, role, content, **kwargs):
        message = {"role": role, "content": content}

        if "tool_call_id" in kwargs:
            message["tool_call_id"] = kwargs["tool_call_id"]
        if "tool_calls" in kwargs:
            message["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                for tc in kwargs["tool_calls"]
            ]

        if role == "system":
            self._system_message = message
            return

        self._messages.append(message)
        while len(self._messages) > self.max_messages:
            self._messages.pop(0)
```

Note the `tool_calls` serialization: arguments go back to a JSON string here because that is what the OpenAI API expects in the message history. You parsed them into a dict when reading, and serialize them back when storing.

```python
    def get_messages(self):
        result = []
        if self._system_message:
            result.append(self._system_message)
        result.extend(self._messages)
        return result
```

Always prepends the system message. This is what gets sent to the LLM on every call.

### `memory/summary.py` — LLM-Compressed Memory

When the buffer exceeds `max_messages`, instead of dropping old messages, it asks the LLM to summarize them:

```python
    def _compress(self):
        split_point = len(self._messages) // 2
        old_messages = self._messages[:split_point]
        recent_messages = self._messages[split_point:]

        # Ask LLM to summarize old messages
        response = self._llm.chat([{"role": "user", "content": prompt}])
        self._summary = response.content.strip()
        self._messages = recent_messages
```

The summary gets injected into the system message on every `get_messages()` call so the LLM always has context about earlier conversation even though those messages are gone.

---

## Module 3: `tools/` — The Tool Engine

This is the most technically interesting module. It converts plain Python functions into JSON Schema objects that the OpenAI API understands.

### `tools/base.py` — The `@tool` Decorator

The core problem: OpenAI's function calling requires tools in this format:

```json
{
  "type": "function",
  "function": {
    "name": "calculate",
    "description": "Evaluate a math expression.",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {"type": "string", "description": "The math expression."}
      },
      "required": ["expression"]
    }
  }
}
```

Writing this by hand for every function is tedious and error-prone. The `@tool` decorator generates it automatically from the function's type hints and docstring.

**Step 1: Type mapping**

```python
_TYPE_MAP: dict[type, str] = {
    str: "string", int: "integer", float: "number",
    bool: "boolean", list: "array", dict: "object",
}
```

Python types map to JSON Schema types.

**Step 2: Parameter schema extraction**

```python
def _extract_parameters_schema(func):
    hints = get_type_hints(func)       # {"expression": str, "return": str}
    sig = inspect.signature(func)      # inspect default values

    for param_name, param in sig.parameters.items():
        py_type = hints.get(param_name, str)
        json_type = _python_type_to_json_schema(py_type)
        prop = {"type": json_type}

        if param.default is inspect.Parameter.empty:
            required.append(param_name)   # no default = required
```

`get_type_hints()` resolves forward references and returns a dict of param name to type. `inspect.signature()` gives access to default values — if a param has no default, it goes in the `required` list.

**Step 3: Docstring parsing**

```python
def _parse_docstring_params(docstring):
    # Parses Google-style Args: blocks
    # "    expression: The math expression." → {"expression": "The math expression."}
```

Scans line by line for `Args:` section, then parses `param_name: description` pairs.

**Step 4: The decorator**

```python
def tool(func):
    name = func.__name__
    description = func.__doc__.strip().split("\n")[0]
    parameters = _extract_parameters_schema(func)

    return BaseTool(name=name, description=description, parameters=parameters, func=func)
```

Returns a `BaseTool` dataclass instance. The original function is stored in `func` and called via `__call__`.

```python
@dataclass
class BaseTool:
    name: str
    description: str
    parameters: dict
    func: Callable

    def __call__(self, **kwargs):
        return self.func(**kwargs)

    def to_openai_schema(self):
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.parameters,
        }}
```

### `tools/registry.py` — Name-Based Lookup

```python
class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool):
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered.")
        self._tools[tool.name] = tool

    def get(self, name) -> BaseTool | None:
        return self._tools.get(name)

    def to_openai_schema(self):
        return [tool.to_openai_schema() for tool in self._tools.values()]
```

A dict from tool name to `BaseTool`. `to_openai_schema()` converts all registered tools to the list format the OpenAI API expects. This list is passed on every `chat_with_tools()` call.

### `tools/executor.py` — Dispatching Tool Calls

```python
class ToolExecutor:
    def execute(self, tool_call: dict) -> str:
        name = tool_call["name"]
        arguments = tool_call["arguments"]

        tool = self.registry.get(name)
        if tool is None:
            return f"Error: Tool '{name}' not found."

        try:
            result = tool(**arguments)
            return str(result) if result is not None else "Done (no output)"
        except Exception as e:
            return f"Error: Tool '{name}' failed: {e}"
```

Takes the parsed tool call dict from `LLMResponse.tool_calls`, looks up the tool, calls it with `**arguments`, returns the result as a string. Errors are caught and returned as strings — they do not crash the loop. The loop feeds the error back to the LLM as an observation so it can recover.

```python
    def execute_many(self, tool_calls):
        return [
            {"tool_call_id": tc["id"], "content": self.execute(tc)}
            for tc in tool_calls
        ]
```

Returns a list of `{"tool_call_id": ..., "content": ...}` dicts ready to be added to memory as tool messages.

---

## Module 4: `planning/` — Structuring LLM Reasoning

### Why it exists

Without structure, the LLM just returns free text. Planning forces it to output reasoning in a predictable format that the loop can parse and act on.

### `planning/base.py`

```python
@dataclass
class Step:
    thought: str       # Why am I doing this?
    action: str        # Which tool? Or "Final Answer"
    action_input: str  # What to pass to the tool
```

A `Step` is one unit of reasoning. The loop executes one step at a time.

```python
class BasePlanner(ABC):
    @abstractmethod
    def plan(self, goal: str, context: str = "", available_tools: list[str] | None = None) -> Step: ...
```

### `planning/react.py` — The ReAct Pattern

ReAct (Reasoning + Acting) is a prompting technique from a 2022 paper by Yao et al. The idea: force the LLM to alternate between thinking and acting in a structured format.

**The system prompt:**

```
You have access to the following tools: {tool_descriptions}

ALWAYS use this EXACT format:

Thought: [your reasoning about what to do next]
Action: [the tool name, or "Final Answer"]
Action Input: [the input to the tool, or your final answer]
```

This is pure prompt engineering. The LLM is constrained to output structured text that can be parsed.

**The plan method:**

```python
def plan(self, goal, context="", available_tools=None):
    tools_str = ", ".join(available_tools or [])
    system_prompt = _REACT_SYSTEM_PROMPT.format(tool_descriptions=tools_str)

    user_content = f"Goal: {goal}"
    if context:
        user_content += f"\n\nContext from previous steps:\n{context}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    response = self._llm.chat(messages)
    return self._parse_response(response.content)
```

Each call to `plan()` is a fresh LLM call. The `context` parameter carries the accumulated observations from previous steps so the LLM knows what has already happened.

**The parser:**

```python
@staticmethod
def _parse_response(text) -> Step:
    thought = action = action_input = ""
    for line in text.strip().split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("thought:"):
            thought = stripped[len("thought:"):].strip()
        elif stripped.lower().startswith("action input:"):
            action_input = stripped[len("action input:"):].strip()
        elif stripped.lower().startswith("action:"):
            action = stripped[len("action:"):].strip()

    if not action:
        return Step(thought=thought or text, action="Final Answer", action_input=text)

    return Step(thought=thought, action=action, action_input=action_input)
```

Line-by-line scan for known prefixes. Note `action input:` is checked before `action:` to avoid the shorter prefix matching first. If parsing fails entirely, the whole response is treated as a final answer — a safe fallback.

---

## Module 5: `loop/engine.py` — The Heart of Everything

This is the most important file in the entire codebase. Every agentic framework — LangChain's `AgentExecutor`, CrewAI, AutoGPT — has this loop at its core.

### The fundamental insight

An LLM by itself can only generate text. To make it take actions in the world, you need a loop:

1. Show the LLM the conversation + available tools
2. LLM either answers or requests a tool call
3. If tool call: execute it, add result to conversation, go back to step 1
4. If answer: return it

That's it. Everything else is scaffolding around this loop.

### `AgentLoop.__init__`

```python
def __init__(self, llm, tools=None, memory=None, max_iterations=10,
             system_prompt="You are a helpful AI assistant.", verbose=True):
    self.llm = llm
    self.max_iterations = max_iterations
    self.memory = memory or BufferMemory(max_messages=50)

    self.registry = ToolRegistry()
    if tools:
        self.registry.register_many(tools)
    self.executor = ToolExecutor(self.registry, self.logger)

    self.memory.add_message("system", system_prompt)
```

On init: memory is created (or passed in), tools are registered, system prompt is added to memory. The system prompt is the first message in every conversation.

### `AgentLoop.run` — The Loop

```python
def run(self, user_input: str) -> str:
    self.memory.add_message("user", user_input)
    tool_schemas = self.registry.to_openai_schema()

    for iteration in range(1, self.max_iterations + 1):

        messages = self.memory.get_messages()

        if tool_schemas:
            response = self.llm.chat_with_tools(messages, tool_schemas)
        else:
            response = self.llm.chat(messages)

        if response.has_tool_calls:
            # Record the assistant's tool-call intent in memory
            self.memory.add_assistant_tool_calls(
                content=response.content,
                tool_calls=response.tool_calls,
            )

            # Execute every tool the LLM requested
            results = self.executor.execute_many(response.tool_calls)

            # Add each tool result to memory
            for result in results:
                self.memory.add_tool_message(
                    tool_call_id=result["tool_call_id"],
                    content=result["content"],
                )

            continue  # go back to the LLM with updated context

        # No tool calls = final answer
        final_answer = response.content
        self.memory.add_message("assistant", final_answer)
        return final_answer

    return f"Max iterations ({self.max_iterations}) reached."
```

**Why `add_assistant_tool_calls` before the tool results?**

The OpenAI API enforces a strict message ordering rule. When the LLM requests tool calls, the conversation history must look like this:

```
[user message]
[assistant message WITH tool_calls field]   ← must come first
[tool message with tool_call_id]            ← then the result
[assistant message]                         ← then the next LLM response
```

If you skip the assistant message with `tool_calls` and just add the tool result, the API throws a validation error. This is the most common bug people hit when building tool-calling agents from scratch.

**Why `continue` instead of recursion?**

Recursion would work but risks stack overflow on long chains. The `for` loop with `continue` is flat and bounded by `max_iterations`.

**The `tool_schemas` are computed once before the loop**, not on every iteration. The available tools don't change mid-conversation, so there's no reason to recompute them.

---

## Module 6: `agents/` — Agent Abstractions

### `agents/base.py`

```python
class BaseAgent(ABC):
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description or f"Agent: {name}"

    @abstractmethod
    def run(self, input_text: str) -> str: ...
```

Every agent has a `name`, a `description` (used by other agents to know what this agent does), and a `run()` method. That's the entire contract.

### `agents/single.py` — SingleAgent

```python
class SingleAgent(BaseAgent):
    def __init__(self, name, llm, tools=None, memory=None,
                 system_prompt="You are a helpful AI assistant.",
                 max_iterations=10, verbose=True):
        super().__init__(name=name)
        self._loop = AgentLoop(
            llm=llm, tools=tools, memory=memory,
            max_iterations=max_iterations,
            system_prompt=system_prompt, verbose=verbose,
        )

    def run(self, input_text):
        return self._loop.run(input_text)
```

`SingleAgent` is a thin wrapper around `AgentLoop`. It adds a name, description, and persona. The loop does all the actual work. This is the building block for multi-agent systems.

### `agents/sequential.py` — SequentialPipeline

```python
class SequentialPipeline(BaseAgent):
    def run(self, input_text):
        current_input = input_text

        for i, agent in enumerate(self.agents, 1):
            if i > 1:
                formatted_input = (
                    f"You are step {i} in a pipeline. "
                    f"Here is the output from the previous step:\n\n{current_input}\n\n"
                    f"Original task: {input_text}\n"
                    f"Your role ({agent.name}): {agent.description}"
                )
            else:
                formatted_input = current_input

            current_input = agent.run(formatted_input)

        return current_input
```

Each agent's output becomes the next agent's input. For agents after the first, the input is reformatted to include context about the pipeline stage, the original task, and the agent's role. This prevents agents from losing track of the original goal.

### `agents/crew.py` — CrewAgent (Hierarchical)

Three-phase execution:

**Phase 1: Manager plans delegations**

```python
def _get_delegations(self, task):
    # Manager LLM sees all agent names + descriptions
    # Returns JSON: [{"agent": "Researcher", "task": "Find X"}, ...]
```

The manager LLM is given a system prompt listing all specialist agents and their descriptions. It responds with a JSON array of delegations. If JSON parsing fails, it falls back to broadcasting the task to all agents.

**Phase 2: Specialists execute**

```python
for delegation in delegations:
    agent = self.agents[delegation["agent"]]
    result = agent.run(delegation["task"])
    results.append({"agent": agent_name, "result": result})
```

Each specialist runs independently. Their results are collected.

**Phase 3: Manager synthesizes**

```python
def _synthesize(self, original_task, results):
    # Manager LLM sees all results and produces one coherent answer
```

The manager sees all specialist outputs and combines them into a final answer.

---

## Module 7: `utils/logger.py` — Colored Tracing

```python
class AgentLogger:
    def thought(self, message):   self._log("THOUGHT", message)   # yellow
    def action(self, name, args): self._log("ACTION", ...)         # magenta
    def observation(self, result):self._log("OBSERVATION", ...)    # blue
    def result(self, message):    self._log("RESULT", message)     # green
    def error(self, message):     self._log("ERROR", message)      # red
```

ANSI escape codes for terminal colors. Each log type has a semantic color. When `verbose=False`, `_log()` returns immediately without printing. This is how you silence agents in production or tests.

---

## How All Dependencies Flow Together

This is the dependency graph. Each arrow means "depends on / uses":

```
OpenAILLM
    └── used by → AgentLoop
                      ├── uses → ToolRegistry
                      │               └── holds → BaseTool (created by @tool)
                      ├── uses → ToolExecutor
                      │               └── uses → ToolRegistry
                      ├── uses → BaseMemory (BufferMemory / SummaryMemory)
                      └── used by → SingleAgent
                                        ├── used by → SequentialPipeline
                                        └── used by → CrewAgent

ReActPlanner
    └── uses → BaseLLM directly (separate from AgentLoop)
    └── produces → Step objects
    └── used manually in examples/03_react_agent.py
```

### Dependency resolution at runtime

When you write:

```python
agent = SingleAgent(name="Bot", llm=llm, tools=[calculate])
agent.run("What is 25 * 37?")
```

Here is exactly what happens in order:

1. `SingleAgent.__init__` creates an `AgentLoop`
2. `AgentLoop.__init__` creates a `ToolRegistry`, registers `calculate`, creates a `ToolExecutor`, creates `BufferMemory`, adds system prompt to memory
3. `agent.run("What is 25 * 37?")` calls `AgentLoop.run()`
4. Memory adds user message: `[system, user]`
5. `registry.to_openai_schema()` converts `calculate` to OpenAI function format
6. `llm.chat_with_tools([system, user], [calculate_schema])` → OpenAI API call
7. API returns `tool_calls: [{"id": "call_abc", "name": "calculate", "arguments": "{\"expression\": \"25 * 37\"}"}]`
8. `OpenAILLM` parses: `json.loads(arguments)` → `{"expression": "25 * 37"}`
9. `LLMResponse(tool_calls=[{"id": "call_abc", "name": "calculate", "arguments": {"expression": "25 * 37"}}])`
10. Loop detects `has_tool_calls = True`
11. Memory adds assistant message with `tool_calls` field
12. `executor.execute_many(tool_calls)` → `registry.get("calculate")` → `calculate(expression="25 * 37")` → `"925"`
13. Memory adds tool message: `{"role": "tool", "tool_call_id": "call_abc", "content": "925"}`
14. Loop continues, memory now: `[system, user, assistant+tool_calls, tool_result]`
15. `llm.chat_with_tools(messages, schemas)` → API sees tool result → returns `"25 * 37 equals 925"`
16. `has_tool_calls = False` → final answer → return `"25 * 37 equals 925"`

---

## Common Interview Questions

**Q: Why does the assistant message with tool_calls have to be added to memory before the tool result?**

The OpenAI API validates conversation structure. A `tool` role message must always be preceded by an `assistant` message that contains the corresponding `tool_calls` entry with a matching `id`. If you add the tool result without the assistant message first, the API returns a 400 error. This is the most common bug when building tool-calling agents from scratch.

**Q: Why does `_extract_parameters_schema` use `get_type_hints()` instead of `func.__annotations__`?**

`func.__annotations__` returns raw annotation objects which may be strings (forward references) when `from __future__ import annotations` is used. `get_type_hints()` resolves those strings into actual types. Since the codebase uses `from __future__ import annotations` everywhere, `get_type_hints()` is required.

**Q: How does the ReAct planner differ from the AgentLoop?**

`AgentLoop` uses OpenAI's native function calling — the LLM returns structured `tool_calls` JSON. `ReActPlanner` uses prompt engineering — the LLM returns free text in `Thought/Action/Action Input` format which is then parsed. ReAct works with any LLM that can follow instructions. Native function calling is more reliable but requires API support.

**Q: What happens if a tool throws an exception?**

`ToolExecutor.execute()` wraps the call in a try/except. The exception message is returned as a string: `"Error: Tool 'calculate' failed: ZeroDivisionError: division by zero"`. This string gets added to memory as a tool result. The LLM sees it on the next iteration and can decide to retry with different arguments or give up.

**Q: How does `SummaryMemory` avoid losing context when the buffer fills up?**

When `len(self._messages) > max_messages`, it splits the message list in half. The older half is summarized by the LLM using a compression prompt. The summary is stored as a string. On every `get_messages()` call, the summary is injected into the system message content. The LLM always sees the summary even though the original messages are gone.

**Q: How does `CrewAgent` know which specialist to call?**

The manager LLM is given a system prompt listing all specialist agents with their names and descriptions. It responds with a JSON array of delegations. The `description` field on `BaseAgent` is specifically designed for this — it tells other agents what a given agent is good at. If the manager's JSON is malformed, the fallback is to send the task to all agents.

**Q: Why is `ToolRegistry` a separate class from `ToolExecutor`?**

Single responsibility. The registry is a lookup table — it only knows about tool schemas and name-to-tool mapping. The executor is a dispatcher — it knows how to call tools and handle errors. The loop uses the registry to get schemas for the LLM, and the executor to run tool calls. They could be merged but separating them makes each easier to test and reason about.

**Q: What is `tool_choice = "auto"` in `chat_with_tools`?**

It tells OpenAI the LLM can decide whether to call a tool or respond with text. The alternatives are `"none"` (never call tools) and `{"type": "function", "function": {"name": "..."}}` (force a specific tool). `"auto"` is the right default for agents because you want the LLM to decide when tools are needed.

---

---

## Module 8: `llm/router.py` — SmartRouter (Auto Model Switching)

### Why it exists

No other agent framework auto-routes between cheap and expensive models per-request. SmartRouter pools multiple LLMs and picks the best one based on task characteristics — saving cost on simple queries, upgrading for complex reasoning.

### Usage

```python
from lightagentx import SmartRouter, OpenAILLM, SingleAgent

router = SmartRouter(
    models=[
        OpenAILLM(model="gpt-4o-mini"),    # cheap, fast
        OpenAILLM(model="gpt-4o"),          # powerful, expensive
    ],
    strategy="complexity",  # auto-detect task complexity
)

agent = SingleAgent(name="Bot", llm=router)
agent.run("Hi")                    # → gpt-4o-mini (simple)
agent.run("Analyze the trade-offs of microservices vs monoliths step by step")  # → gpt-4o (complex)
```

### Strategies

| Strategy | Behavior |
|---|---|
| `"complexity"` | Scores messages by length + keywords + depth → picks model tier |
| `"round_robin"` | Cycles through models evenly |
| `"fallback"` | Tries cheapest first, falls back on error |
| `"cost_limit"` | Uses cheap model for first N calls, then upgrades |

### Complexity Heuristics

Three factors scored 0.0–1.0:
- **Length** (0–0.3): Messages >500 chars score higher
- **Keywords** (0–0.4): "analyze", "step by step", "compare", "architecture", etc.
- **Depth** (0–0.3): Conversations >20 messages score higher

### Routing History

Every call is logged for transparency:

```python
for entry in router.routing_history:
    print(f"Call #{entry['call_number']}: {entry['model']} ({entry['strategy']})")
```

### How it works internally

`SmartRouter` extends `BaseLLM` — it **is** an LLM from the framework's perspective. This means it drops into any place that accepts `BaseLLM`: `SingleAgent`, `AgentLoop`, `CrewAgent`, etc. No code changes needed.

```python
class SmartRouter(BaseLLM):
    def chat(self, messages):
        selected = self._select_model(messages)  # pick based on strategy
        return self._execute(selected, "chat", messages)
```

For `"fallback"`, the execute path tries each model in order:

```python
def _execute_with_fallback(self, method, messages, tools, start):
    for model in self.models:
        try:
            return self._call_model(model, method, messages, tools)
        except Exception:
            continue  # try next
    raise RuntimeError("All models failed")
```

---

## Module 9: `snapshot.py` — Agent Snapshots (Portable Agents)

### Why it exists

Build an agent on one project, export it with full memory/config, import it on another project. Like Docker for agents. No other framework offers this.

### Usage

```python
from lightagentx import SingleAgent, AgentSnapshot, OpenAILLM, tool

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

# Build and use on Project A
llm = OpenAILLM()
agent = SingleAgent(name="Researcher", llm=llm, tools=[search],
                    system_prompt="You are a research assistant.")
agent.run("Find info about quantum computing")

# Export full state
agent.snapshot("researcher.agent.json")

# Export persona only (no memory — safe to share)
AgentSnapshot.export_portable(agent, "researcher_card.json")

# On Project B — restore with full context
restored = SingleAgent.from_snapshot("researcher.agent.json", llm=OpenAILLM(), tools=[search])
restored.run("Continue our research...")  # remembers everything
```

### What gets saved

```json
{
  "format_version": 1,
  "timestamp": 1696012800.0,
  "agent": {
    "name": "Researcher",
    "description": "...",
    "system_prompt": "You are a research assistant.",
    "max_iterations": 10
  },
  "llm_config": {
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 1024
  },
  "tool_manifest": [
    {"name": "search", "description": "...", "parameters": {...}}
  ],
  "memory": {
    "type": "BufferMemory",
    "messages": [...]
  }
}
```

**API keys are never saved.** LLM config records model/temperature/max_tokens. You provide a fresh `llm` instance on load.

### Tool validation

On load, the snapshot checks if all expected tools are provided:

```python
restored = AgentSnapshot.load("agent.json", llm=llm, tools=[])
# UserWarning: Snapshot expects tools {'search'} but they were not provided.
```

### Programmatic API

```python
data = AgentSnapshot.to_dict(agent)              # dict
agent = AgentSnapshot.from_dict(data, llm, tools) # restore from dict
```

---

## Module 10: `hooks.py` — Lifecycle Hooks (Event-Driven Middleware)

### Why it exists

Register callbacks for any point in the agent lifecycle without monkey-patching. Enables custom logging, guardrails, cost tracking, rate limiting, and more.

### Usage

```python
from lightagentx import SingleAgent, HookRegistry, OpenAILLM

hooks = HookRegistry()

@hooks.on("before_llm_call")
def log_prompt(event):
    print(f"Sending {len(event.data['messages'])} messages to {event.data['model']}")

@hooks.on("after_tool_call")
def track_tools(event):
    print(f"Tool {event.data['tool_name']} returned: {event.data['result'][:100]}")

@hooks.on("on_error")
def alert(event):
    send_slack_alert(f"Agent error: {event.data['error']}")

agent = SingleAgent(name="Bot", llm=OpenAILLM(), hooks=hooks)
agent.run("Hello")  # hooks fire automatically
```

### Supported Events

| Event | Fires when | Data keys |
|---|---|---|
| `before_llm_call` | Before each LLM API call | `messages`, `model` |
| `after_llm_call` | After each LLM response | `messages`, `model`, `response` |
| `before_tool_call` | Before each tool execution | `tool_name`, `arguments` |
| `after_tool_call` | After each tool returns | `tool_name`, `arguments`, `result` |
| `on_iteration` | Each loop iteration | `iteration`, `max_iterations` |
| `on_error` | Any exception in the loop | `error`, `context` |
| `on_agent_start` | `agent.run()` begins | `agent_name`, `input_text` |
| `on_agent_end` | `agent.run()` completes | `agent_name`, `input_text`, `output` |

### Safety

Hook errors **never crash the agent**. Exceptions in callbacks are silently caught:

```python
def emit(self, event_name, **data):
    for callback in self._hooks[event_name]:
        try:
            callback(event)
        except Exception:
            pass  # hooks must never crash the agent
```

### Programmatic registration

```python
hooks.register("before_llm_call", my_callback)
hooks.clear("before_llm_call")  # clear specific event
hooks.clear()                    # clear all hooks
print(hooks.registered_events)   # ['after_tool_call', ...]
```

---

## LangChain Equivalence Map

| LightAgentX | LangChain |
|---|---|
| `BaseLLM` / `OpenAILLM` | `BaseChatModel` / `ChatOpenAI` |
| `SmartRouter` | No equivalent (custom routing needed) |
| `LLMResponse` | `AIMessage` |
| `BufferMemory` | `ConversationBufferWindowMemory` |
| `SummaryMemory` | `ConversationSummaryMemory` |
| `@tool` / `BaseTool` | `@tool` / `StructuredTool` |
| `ToolRegistry.to_openai_schema()` | `format_tool_to_openai_function()` |
| `AgentLoop` | `AgentExecutor.invoke()` |
| `ReActPlanner` | `create_react_agent` + `ReActOutputParser` |
| `SingleAgent` | `create_tool_calling_agent` + `AgentExecutor` |
| `SequentialPipeline` | `SequentialChain` / LCEL `chain1 \| chain2` |
| `CrewAgent` | CrewAI `Crew(process=Process.hierarchical)` |
| `HookRegistry` | LangChain Callbacks / `BaseCallbackHandler` |
| `AgentSnapshot` | No equivalent (manual serialization needed) |
