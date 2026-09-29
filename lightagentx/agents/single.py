"""Single Agent — one LLM + tools + memory wrapped with a persona."""

from __future__ import annotations

from typing import Any

from .base import BaseAgent
from ..llm.base import BaseLLM
from ..memory.base import BaseMemory
from ..memory.buffer import BufferMemory
from ..loop.engine import AgentLoop
from ..utils.logger import AgentLogger


class SingleAgent(BaseAgent):
    """
    A single agent powered by one LLM, with optional tools and memory.

    Usage:
        agent = SingleAgent(
            name="Calculator Bot",
            llm=OpenAILLM(),
            tools=[calculate_tool],
            system_prompt="You are a math expert.",
        )
        result = agent.run("What is 25 * 37?")
    """

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        tools: list[Any] | None = None,
        memory: BaseMemory | None = None,
        system_prompt: str = "You are a helpful AI assistant.",
        description: str = "",
        max_iterations: int = 10,
        verbose: bool = True,
    ):
        super().__init__(name=name, description=description)
        self.llm = llm
        self.tools = tools or []
        self.memory = memory or BufferMemory(max_messages=50)
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.logger = AgentLogger(verbose=verbose)

        self._loop = AgentLoop(
            llm=self.llm,
            tools=self.tools,
            memory=self.memory,
            max_iterations=self.max_iterations,
            system_prompt=self.system_prompt,
            verbose=self.verbose,
        )

    def run(self, input_text: str) -> str:
        """Run the agent on the given input."""
        self.logger.agent(self.name, f"Starting task: {input_text[:100]}")
        result = self._loop.run(input_text)
        self.logger.agent(self.name, "Task complete")
        return result

    def reset(self) -> None:
        """Reset the agent's memory and loop state."""
        self._loop.reset()
        self.memory = BufferMemory(max_messages=50)
        self._loop = AgentLoop(
            llm=self.llm,
            tools=self.tools,
            memory=self.memory,
            max_iterations=self.max_iterations,
            system_prompt=self.system_prompt,
            verbose=self.verbose,
        )
