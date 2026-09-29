"""
ReAct Planner — implements the Reasoning + Acting pattern.

HOW THIS MAPS TO LANGCHAIN:
  - This is the core of LangChain's ReAct agent (`create_react_agent`)
  - The prompt forces the LLM to output structured Thought/Action/Action Input
  - LangChain uses an `OutputParser` to parse this; we do it with simple string parsing

THE REACT PATTERN (Yao et al., 2022):
  The LLM alternates between:
    Thought: reasoning about the current state
    Action: choosing a tool to use (or "Final Answer")
    Action Input: the input to that tool
    Observation: (filled in by the tool result)

  This continues until the LLM outputs "Final Answer" as the action.

WHAT YOU LEARN HERE:
  1. How to craft prompts that force structured LLM output
  2. How to parse semi-structured text from an LLM
  3. The fundamental ReAct loop that powers most LangChain agents
"""

from __future__ import annotations

from typing import Any

from .base import BasePlanner, Step


_REACT_SYSTEM_PROMPT = """You are a reasoning agent. You solve problems by thinking step-by-step and using tools.

You have access to the following tools:
{tool_descriptions}

ALWAYS use this EXACT format for your response:

Thought: [your reasoning about what to do next]
Action: [the tool name to use, or "Final Answer" if you have the answer]
Action Input: [the input to the tool, or your final answer text]

Important rules:
- ALWAYS start with "Thought:"
- ALWAYS follow with "Action:" and "Action Input:"
- Use "Final Answer" as the Action when you have enough information to answer
- Use EXACTLY one Thought/Action/Action Input block per response
- Do NOT include "Observation:" — that will be provided by the system
"""


class ReActPlanner(BasePlanner):
    """
    Planner that uses the ReAct pattern to decide the next action.

    It constructs a prompt, sends it to the LLM, and parses the structured
    Thought/Action/Action Input response.

    Usage:
        planner = ReActPlanner(llm)
        step = planner.plan(
            goal="What's 25 * 4?",
            context="",
            available_tools=["calculate"]
        )
        print(step.thought)       # "I need to calculate 25 * 4"
        print(step.action)        # "calculate"
        print(step.action_input)  # "25 * 4"
    """

    def __init__(self, llm: Any):
        """
        Args:
            llm: A BaseLLM instance used for reasoning.
        """
        self._llm = llm

    def plan(
        self,
        goal: str,
        context: str = "",
        available_tools: list[str] | None = None,
    ) -> Step:
        """
        Ask the LLM to reason and produce the next step.

        The method:
        1. Builds a system prompt listing available tools
        2. Constructs a user message with the goal + any prior context
        3. Calls the LLM
        4. Parses the response into a Step
        """
        tools_str = ", ".join(available_tools or [])
        system_prompt = _REACT_SYSTEM_PROMPT.format(tool_descriptions=tools_str)

        # Build the user message
        user_content = f"Goal: {goal}"
        if context:
            user_content += f"\n\nContext from previous steps:\n{context}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = self._llm.chat(messages)
        return self._parse_response(response.content)

    @staticmethod
    def _parse_response(text: str) -> Step:
        """
        Parse the LLM's ReAct-formatted response into a Step.

        Expected format:
            Thought: ...
            Action: ...
            Action Input: ...

        HOW THIS WORKS:
          We scan line by line, looking for the prefixes "Thought:", "Action:",
          and "Action Input:". This is simple but effective — the same basic
          approach LangChain uses in its ReAct output parser.
        """
        thought = ""
        action = ""
        action_input = ""

        lines = text.strip().split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("thought:"):
                thought = stripped[len("thought:"):].strip()
            elif stripped.lower().startswith("action input:"):
                action_input = stripped[len("action input:"):].strip()
            elif stripped.lower().startswith("action:"):
                action = stripped[len("action:"):].strip()

        # Fallback: if parsing failed, treat the whole response as a final answer
        if not action:
            return Step(
                thought=thought or text,
                action="Final Answer",
                action_input=text,
            )

        return Step(thought=thought, action=action, action_input=action_input)
