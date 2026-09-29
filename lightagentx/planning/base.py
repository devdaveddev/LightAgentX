"""
Abstract base class for task planners.

HOW THIS MAPS TO LANGCHAIN:
  - LangChain doesn't have a separate "planner" abstraction — it's baked
    into the agent prompt and output parser
  - We make it explicit so you understand the planning step clearly

WHAT YOU LEARN HERE:
  A planner takes a goal + context and produces a sequence of Steps.
  Each Step has a thought (reasoning) and an action (what to do).
  The loop engine then executes these steps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Step:
    """
    A single step in a plan.

    Attributes:
        thought: The reasoning behind this step.
        action: The tool/action to take (or "Final Answer").
        action_input: The input to the action.
    """

    thought: str
    action: str
    action_input: str

    def __repr__(self) -> str:
        return f"Step(action={self.action!r}, input={self.action_input[:50]!r})"


class BasePlanner(ABC):
    """
    Abstract base for planners.

    A planner analyzes the goal and available tools, then outputs
    one or more Steps for the agent loop to execute.
    """

    @abstractmethod
    def plan(
        self,
        goal: str,
        context: str = "",
        available_tools: list[str] | None = None,
    ) -> Step:
        """
        Produce the next step given a goal and current context.

        Args:
            goal: What the agent is trying to achieve.
            context: Current conversation/observation context.
            available_tools: Names of tools the agent can use.

        Returns:
            The next Step to execute.
        """
        ...
