"""
Single Agent — one LLM + tools + memory wrapped with a persona.

HOW THIS MAPS TO LANGCHAIN:
  - This combines LangChain's `create_tool_calling_agent()` + `AgentExecutor`
  - In CrewAI, this is a single `Agent` with a `role` and `goal`

WHAT YOU LEARN HERE:
  A SingleAgent is essentially a configured AgentLoop with a personality.
  It's the building block for multi-agent systems.
"""

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

        # The AgentLoop does the actual work
        self._loop = AgentLoop(
            llm=self.llm,
            tools=self.tools,
            memory=self.memory,
            max_iterations=self.max_iterations,
            system_prompt=self.system_prompt,
            verbose=self.verbose,
        )

    def run(self, input_text: str) -> str:
        """
        Run the agent on the given input.

        Delegates to the AgentLoop which handles the full
        reasoning → tool-calling → observation cycle.
        """
        self.logger.agent(self.name, f"Starting task: {input_text[:100]}")
        result = self._loop.run(input_text)
        self.logger.agent(self.name, "Task complete")
        return result

    def reset(self) -> None:
        """Reset the agent's memory and loop state."""
        self._loop.reset()
        # Reconstruct the loop with a fresh memory
        self.memory = BufferMemory(max_messages=50)
        self._loop = AgentLoop(
            llm=self.llm,
            tools=self.tools,
            memory=self.memory,
            max_iterations=self.max_iterations,
            system_prompt=self.system_prompt,
            verbose=self.verbose,
        )
