"""
Abstract base class for agents.

HOW THIS MAPS TO LANGCHAIN:
  - LangChain's `BaseChain` / `Runnable` serves a similar role
  - Every agent is something you can `.run(input)` and get a string back
  - This simple interface enables composability (pipelines, crews)

WHAT YOU LEARN HERE:
  The agent abstraction is deliberately simple — just `run(input) → output`.
  This makes it trivial to chain agents together.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Abstract base for all agents.

    Every agent has a name, a description (so other agents know what it does),
    and a `run()` method.
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description or f"Agent: {name}"

    @abstractmethod
    def run(self, input_text: str) -> str:
        """
        Run the agent with the given input and return the output.

        Args:
            input_text: The task or question for the agent.

        Returns:
            The agent's response as a string.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
