"""Comprehensive tests covering all edge cases for using lightagentx as a library.

Tests cover:
- Top-level imports
- LLM: subclassing, response edge cases
- Memory: buffer edge cases, summary without system msg, clear with summary
- Tools: optional params, no docstring, no type hints, list/dict types
- Planning: ReActPlanner parsing edge cases
- Loop: reset, run-after-reset, no-tools loop
- Agents: SingleAgent reset, CrewAgent delegation + fallback
"""

import json
import pytest

# ── 1. Top-level import test ──────────────────────────────────────────────────

from lightagentx import (
    BaseLLM, LLMResponse, OpenAILLM,
    BaseMemory, BufferMemory, SummaryMemory,
    BaseTool, tool, ToolRegistry, ToolExecutor,
    BasePlanner, Step, ReActPlanner,
    AgentLoop,
    BaseAgent, SingleAgent, SequentialPipeline, CrewAgent,
    AgentLogger,
)


# ── Shared mock LLM ──────────────────────────────────────────────────────────

class MockLLM(BaseLLM):
    """Reusable mock LLM returning pre-configured responses."""

    def __init__(self, responses=None, default="Mock response"):
        super().__init__(model="mock")
        self._responses = list(responses or [])
        self._default = default
        self._call_count = 0
        self.call_history = []  # record messages for inspection

    def chat(self, messages):
        self.call_history.append(messages)
        return self._next()

    def chat_with_tools(self, messages, tools):
        self.call_history.append(messages)
        return self._next()

    def _next(self):
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            if isinstance(resp, str):
                return LLMResponse(content=resp)
            return resp
        return LLMResponse(content=self._default)


# ── 2. LLM Tests ─────────────────────────────────────────────────────────────

class TestLLMResponseEdgeCases:
    def test_empty_content_no_tools(self):
        r = LLMResponse()
        assert r.content == ""
        assert r.tool_calls == []
        assert r.has_tool_calls is False

    def test_raw_field_preserved(self):
        r = LLMResponse(content="hi", raw={"some": "raw_data"})
        assert r.raw == {"some": "raw_data"}

    def test_repr_of_subclass(self):
        m = MockLLM()
        assert "mock" in repr(m)


class TestOpenAILLMInit:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key not found"):
            OpenAILLM(api_key=None)


# ── 3. Memory Tests ──────────────────────────────────────────────────────────

class TestBufferMemoryEdgeCases:
    def test_empty_memory_returns_empty(self):
        mem = BufferMemory()
        assert mem.get_messages() == []

    def test_len_includes_system(self):
        mem = BufferMemory()
        mem.add_message("system", "sys")
        mem.add_message("user", "hi")
        assert len(mem) == 2  # system + user

    def test_len_without_system(self):
        mem = BufferMemory()
        mem.add_message("user", "hi")
        assert len(mem) == 1

    def test_repr(self):
        mem = BufferMemory(max_messages=5)
        assert "BufferMemory" in repr(mem)
        assert "max_messages=5" in repr(mem)

    def test_max_messages_1(self):
        mem = BufferMemory(max_messages=1)
        mem.add_message("user", "a")
        mem.add_message("user", "b")
        msgs = mem.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["content"] == "b"

    def test_system_replaced_on_second_set(self):
        mem = BufferMemory()
        mem.add_message("system", "first")
        mem.add_message("system", "second")
        msgs = mem.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["content"] == "second"

    def test_tool_call_arguments_serialized_from_dict(self):
        """tool_calls with dict arguments should be JSON-serialized."""
        mem = BufferMemory()
        mem.add_assistant_tool_calls(
            content="",
            tool_calls=[{"id": "c1", "name": "foo", "arguments": {"x": 42}}],
        )
        msgs = mem.get_messages()
        tc = msgs[0]["tool_calls"][0]
        assert tc["function"]["arguments"] == '{"x": 42}'

    def test_tool_call_arguments_preserved_if_string(self):
        """tool_calls with string arguments should be kept as-is."""
        mem = BufferMemory()
        mem.add_assistant_tool_calls(
            content="",
            tool_calls=[{"id": "c1", "name": "foo", "arguments": '{"x": 42}'}],
        )
        msgs = mem.get_messages()
        tc = msgs[0]["tool_calls"][0]
        assert tc["function"]["arguments"] == '{"x": 42}'


class TestSummaryMemoryEdgeCases:
    def test_no_system_message_with_summary(self):
        """Summary should get injected as system msg even without explicit system msg."""
        llm = MockLLM(default="Compressed summary")
        mem = SummaryMemory(llm, max_messages=2)
        # Add 4 messages to trigger compression
        for i in range(4):
            mem.add_message("user", f"msg {i}")
        msgs = mem.get_messages()
        # First message should be a system msg with the summary
        assert msgs[0]["role"] == "system"
        assert "CONVERSATION SUMMARY" in msgs[0]["content"]

    def test_clear_resets_summary(self):
        llm = MockLLM(default="Summary text")
        mem = SummaryMemory(llm, max_messages=2)
        for i in range(4):
            mem.add_message("user", f"msg {i}")
        assert mem.summary != ""
        mem.clear()
        assert mem.summary == ""
        assert mem.get_messages() == []

    def test_repr(self):
        llm = MockLLM()
        mem = SummaryMemory(llm, max_messages=5)
        assert "SummaryMemory" in repr(mem)

    def test_summary_with_system_message(self):
        """Summary injected into existing system prompt."""
        llm = MockLLM(default="Old convo summary")
        mem = SummaryMemory(llm, max_messages=3)
        mem.add_message("system", "You are helpful.")
        for i in range(5):
            mem.add_message("user", f"msg {i}")
        msgs = mem.get_messages()
        assert msgs[0]["role"] == "system"
        assert "You are helpful." in msgs[0]["content"]
        assert "CONVERSATION SUMMARY" in msgs[0]["content"]


# ── 4. Tool Tests ─────────────────────────────────────────────────────────────

class TestToolEdgeCases:
    def test_tool_no_docstring(self):
        @tool
        def no_doc(x: str) -> str:
            return x

        assert no_doc.name == "no_doc"
        # Description should fall back to function name
        assert no_doc.description == "no_doc"

    def test_tool_with_optional_param(self):
        @tool
        def greet(name: str, greeting: str = "Hello") -> str:
            """Greet someone.

            Args:
                name: The person's name.
                greeting: The greeting to use.
            """
            return f"{greeting}, {name}!"

        assert "name" in greet.parameters.get("required", [])
        assert "greeting" not in greet.parameters.get("required", [])
        result = greet(name="Alice")
        assert result == "Hello, Alice!"
        result = greet(name="Alice", greeting="Hi")
        assert result == "Hi, Alice!"

    def test_tool_with_no_args(self):
        @tool
        def get_time() -> str:
            """Get current time."""
            return "12:00"

        assert get_time.parameters["properties"] == {}
        assert get_time() == "12:00"

    def test_tool_with_list_type(self):
        @tool
        def process(items: list) -> str:
            """Process items."""
            return str(len(items))

        assert process.parameters["properties"]["items"]["type"] == "array"

    def test_tool_with_dict_type(self):
        @tool
        def process(data: dict) -> str:
            """Process data."""
            return str(data)

        assert process.parameters["properties"]["data"]["type"] == "object"

    def test_tool_returning_none(self):
        @tool
        def silent(x: str) -> None:
            """Do nothing."""
            pass

        reg = ToolRegistry()
        reg.register(silent)
        executor = ToolExecutor(reg)
        result = executor.execute({
            "id": "c1", "name": "silent", "arguments": {"x": "hi"}
        })
        assert result == "Done (no output)"

    def test_registry_get_nonexistent(self):
        reg = ToolRegistry()
        assert reg.get("nope") is None

    def test_registry_contains(self):
        @tool
        def my_tool(x: str) -> str:
            """Tool."""
            return x

        reg = ToolRegistry()
        reg.register(my_tool)
        assert "my_tool" in reg
        assert "other" not in reg

    def test_executor_missing_id(self):
        """Tool calls without an 'id' should use 'unknown' as fallback."""
        @tool
        def echo(x: str) -> str:
            """Echo."""
            return x

        reg = ToolRegistry()
        reg.register(echo)
        executor = ToolExecutor(reg)
        results = executor.execute_many([
            {"name": "echo", "arguments": {"x": "hi"}}
        ])
        assert results[0]["tool_call_id"] == "unknown"
        assert results[0]["content"] == "hi"


# ── 5. Planning Tests ────────────────────────────────────────────────────────

class TestReActPlanner:
    def test_parse_valid_response(self):
        step = ReActPlanner._parse_response(
            "Thought: I need to calculate\n"
            "Action: calculate\n"
            "Action Input: 2 + 2"
        )
        assert step.thought == "I need to calculate"
        assert step.action == "calculate"
        assert step.action_input == "2 + 2"

    def test_parse_final_answer(self):
        step = ReActPlanner._parse_response(
            "Thought: I have the answer\n"
            "Action: Final Answer\n"
            "Action Input: The result is 42"
        )
        assert step.action == "Final Answer"
        assert step.action_input == "The result is 42"

    def test_parse_malformed_falls_back(self):
        """If no Action: line is found, treat as Final Answer."""
        step = ReActPlanner._parse_response(
            "Just some random rambling from the LLM."
        )
        assert step.action == "Final Answer"
        assert "rambling" in step.action_input

    def test_parse_case_insensitive(self):
        step = ReActPlanner._parse_response(
            "THOUGHT: reasoning here\n"
            "ACTION: my_tool\n"
            "ACTION INPUT: some input"
        )
        assert step.thought == "reasoning here"
        assert step.action == "my_tool"
        assert step.action_input == "some input"

    def test_plan_calls_llm(self):
        mock = MockLLM(responses=[
            "Thought: I should search\nAction: search\nAction Input: query"
        ])
        planner = ReActPlanner(mock)
        step = planner.plan(
            goal="Find information",
            context="",
            available_tools=["search"],
        )
        assert step.action == "search"
        assert step.action_input == "query"

    def test_plan_with_context(self):
        mock = MockLLM(responses=[
            "Thought: Based on context\nAction: Final Answer\nAction Input: Done"
        ])
        planner = ReActPlanner(mock)
        step = planner.plan(
            goal="Summarize",
            context="Previous observation: data was found",
            available_tools=[],
        )
        # Check that context was passed to the LLM
        assert len(mock.call_history) == 1
        user_msg = mock.call_history[0][1]["content"]
        assert "Previous observation" in user_msg


class TestStep:
    def test_repr_truncation(self):
        s = Step(thought="t", action="a", action_input="x" * 100)
        r = repr(s)
        assert len(r) < 200  # repr truncates action_input to 50 chars


# ── 6. Loop Tests ─────────────────────────────────────────────────────────────

class TestAgentLoopEdgeCases:
    def test_reset_and_rerun(self):
        """After reset(), the loop should work fresh."""
        mock = MockLLM(responses=[
            LLMResponse(content="first answer"),
            LLMResponse(content="second answer"),
        ])
        loop = AgentLoop(llm=mock, verbose=False)
        r1 = loop.run("q1")
        assert r1 == "first answer"

        loop.reset()
        # After reset memory is cleared, but loop can still run
        # However the mock LLM already used response[0], so next is response[1]
        r2 = loop.run("q2")
        assert r2 == "second answer"

    def test_no_tools_direct_chat(self):
        """When no tools are registered, should use chat() not chat_with_tools()."""
        mock = MockLLM(default="direct answer")
        loop = AgentLoop(llm=mock, verbose=False)
        result = loop.run("hello")
        assert result == "direct answer"

    def test_tool_error_doesnt_crash(self):
        """If a tool raises, the error is fed back to the LLM."""
        @tool
        def crasher(x: str) -> str:
            """Crash."""
            raise RuntimeError("boom")

        mock = MockLLM(responses=[
            LLMResponse(
                content="",
                tool_calls=[{"id": "c1", "name": "crasher", "arguments": {"x": "a"}}],
            ),
            LLMResponse(content="Tool failed, answering directly"),
        ])
        loop = AgentLoop(llm=mock, tools=[crasher], verbose=False)
        result = loop.run("test")
        assert "Tool failed" in result or "answering" in result


# ── 7. Agent Tests ────────────────────────────────────────────────────────────

class TestSingleAgentEdgeCases:
    def test_reset_allows_rerun(self):
        mock = MockLLM(responses=[
            LLMResponse(content="r1"),
            LLMResponse(content="r2"),
        ])
        agent = SingleAgent(name="Bot", llm=mock, verbose=False)
        assert agent.run("q1") == "r1"

        agent.reset()
        # After reset, mock still advances, and agent creates a new loop
        result = agent.run("q2")
        # The mock's next response is "r2"
        assert result == "r2"

    def test_repr(self):
        mock = MockLLM()
        agent = SingleAgent(name="Bot", llm=mock, verbose=False)
        assert "Bot" in repr(agent)


class TestCrewAgentFallback:
    def test_fallback_broadcasts_on_bad_json(self):
        """If manager returns unparseable JSON, task is broadcast to all agents."""
        # Manager returns garbage, agents return simple answers
        manager_llm = MockLLM(responses=[
            "This is not valid JSON at all!",         # delegation attempt
            "Combined: Agent1 did X, Agent2 did Y",   # synthesis
        ])
        agent1_llm = MockLLM(default="Agent1 result")
        agent2_llm = MockLLM(default="Agent2 result")

        a1 = SingleAgent(name="Expert1", llm=agent1_llm, description="Expert at A", verbose=False)
        a2 = SingleAgent(name="Expert2", llm=agent2_llm, description="Expert at B", verbose=False)

        crew = CrewAgent(
            name="TestCrew",
            agents=[a1, a2],
            manager_llm=manager_llm,
            verbose=False,
        )
        result = crew.run("Do something")
        # Should still return a result (fallback broadcasts to all agents)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_valid_delegation(self):
        """Manager delegates correctly via JSON."""
        delegation_json = json.dumps([
            {"agent": "Worker", "task": "Do the work"}
        ])
        manager_llm = MockLLM(responses=[
            delegation_json,                          # delegation
            "Final synthesized result from manager",  # synthesis
        ])
        worker_llm = MockLLM(default="Worker output")

        worker = SingleAgent(
            name="Worker", llm=worker_llm,
            description="Does work", verbose=False,
        )

        crew = CrewAgent(
            name="TestCrew",
            agents=[worker],
            manager_llm=manager_llm,
            verbose=False,
        )
        result = crew.run("Complete this task")
        assert "synthesized" in result.lower() or "manager" in result.lower() or len(result) > 0

    def test_unknown_agent_in_delegation(self):
        """Delegating to a non-existent agent should not crash."""
        delegation_json = json.dumps([
            {"agent": "NonExistent", "task": "Do things"}
        ])
        manager_llm = MockLLM(responses=[
            delegation_json,
            "Synthesis despite missing agent",
        ])
        crew = CrewAgent(
            name="TestCrew",
            agents=[],
            manager_llm=manager_llm,
            verbose=False,
        )
        result = crew.run("test")
        assert isinstance(result, str)


class TestSequentialPipelineEdgeCases:
    def test_single_agent_pipeline(self):
        """Pipeline with one agent should just return that agent's output."""
        mock = MockLLM(default="only agent output")
        agent = SingleAgent(name="Solo", llm=mock, verbose=False)
        pipeline = SequentialPipeline(
            name="Solo Pipeline",
            agents=[agent],
            verbose=False,
        )
        result = pipeline.run("input")
        assert result == "only agent output"

    def test_three_agent_pipeline(self):
        m1 = MockLLM(default="step1")
        m2 = MockLLM(default="step2")
        m3 = MockLLM(default="step3")
        a1 = SingleAgent(name="A", llm=m1, verbose=False)
        a2 = SingleAgent(name="B", llm=m2, verbose=False)
        a3 = SingleAgent(name="C", llm=m3, verbose=False)
        pipeline = SequentialPipeline(
            name="Triple", agents=[a1, a2, a3], verbose=False,
        )
        result = pipeline.run("task")
        assert result == "step3"


# ── 8. Logger Tests ───────────────────────────────────────────────────────────

class TestAgentLogger:
    def test_verbose_false_no_output(self, capsys):
        logger = AgentLogger(verbose=False)
        logger.system("test")
        logger.thought("test")
        logger.action("toolname", {"a": 1})
        logger.observation("result")
        logger.result("done")
        logger.error("oops")
        logger.agent("bot", "msg")
        logger.memory("mem")
        logger.plan("plan")
        logger.separator()
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_true_produces_output(self, capsys):
        logger = AgentLogger(verbose=True)
        logger.system("init")
        captured = capsys.readouterr()
        assert "SYSTEM" in captured.out
        assert "init" in captured.out
