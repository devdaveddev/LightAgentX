"""Tests for agent orchestration — SingleAgent, SequentialPipeline, CrewAgent."""

from lightagentx.llm.base import BaseLLM, LLMResponse
from lightagentx.tools.base import tool
from lightagentx.agents.single import SingleAgent
from lightagentx.agents.sequential import SequentialPipeline


class MockLLM(BaseLLM):
    """Mock LLM for testing agents."""

    def __init__(self, responses=None, default_response="Mock response"):
        super().__init__(model="mock")
        self._responses = list(responses or [])
        self._default = default_response
        self._call_count = 0

    def chat(self, messages):
        return self._next()

    def chat_with_tools(self, messages, tools):
        return self._next()

    def _next(self):
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            if isinstance(resp, str):
                return LLMResponse(content=resp)
            return resp
        return LLMResponse(content=self._default)


class TestSingleAgent:
    def test_basic_run(self):
        mock = MockLLM(default_response="Hello from agent!")
        agent = SingleAgent(name="TestBot", llm=mock, verbose=False)
        result = agent.run("Say hello")
        assert result == "Hello from agent!"

    def test_with_tools(self):
        @tool
        def upper(text: str) -> str:
            """Convert to uppercase."""
            return text.upper()

        mock = MockLLM(responses=[
            LLMResponse(
                content="",
                tool_calls=[{
                    "id": "c1",
                    "name": "upper",
                    "arguments": {"text": "hello"},
                }],
            ),
            LLMResponse(content="HELLO"),
        ])

        agent = SingleAgent(name="UpperBot", llm=mock, tools=[upper], verbose=False)
        result = agent.run("Uppercase hello")
        assert result == "HELLO"


class TestSequentialPipeline:
    def test_two_agent_pipeline(self):
        """Output of agent 1 flows to agent 2."""
        mock1 = MockLLM(default_response="Research: AI is transforming coding")
        mock2 = MockLLM(default_response="Article: AI coding assistants are revolutionary")

        agent1 = SingleAgent(name="Researcher", llm=mock1, verbose=False)
        agent2 = SingleAgent(name="Writer", llm=mock2, verbose=False)

        pipeline = SequentialPipeline(
            name="Test Pipeline",
            agents=[agent1, agent2],
            verbose=False,
        )

        result = pipeline.run("Write about AI")
        # The final result should be from the last agent
        assert "Article" in result or "revolutionary" in result
