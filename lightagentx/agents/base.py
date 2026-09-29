"""Abstract base class for agents."""

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
