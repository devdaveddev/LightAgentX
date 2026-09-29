"""
Agent Loop Engine — the heart of the entire framework.

This is THE most important module to understand. It implements the dynamic
tool-calling loop that all agentic AI frameworks use.

HOW THIS MAPS TO LANGCHAIN:
  - This is LangChain's `AgentExecutor.invoke()` method
  - CrewAI's agent execution loop works the same way
  - AutoGPT, BabyAGI — they all have this same core loop

THE LOOP (pseudocode):
    while iterations < max:
        response = llm.chat_with_tools(messages, tools)

        if response has tool_calls:
            for each tool_call:
                result = executor.execute(tool_call)
                messages.append(tool result)
            continue  ← go back to the LLM with new info

        else:
            return response.content  ← final answer!

WHAT YOU LEARN HERE:
  1. The fundamental loop: LLM → decide → execute → observe → repeat
  2. How tool results get fed back into the conversation
  3. How iteration limits prevent infinite loops
  4. How the conversation message format works with tool calls
"""

from __future__ import annotations

from typing import Any

from ..llm.base import BaseLLM
from ..memory.base import BaseMemory
from ..tools.registry import ToolRegistry
from ..tools.executor import ToolExecutor
from ..utils.logger import AgentLogger


class AgentLoop:
    """
    The core reasoning loop that drives tool-calling agents.

    This loop:
    1. Sends the conversation (with tool schemas) to the LLM
    2. If the LLM returns a text response → we're done
    3. If the LLM requests tool calls → execute them, add results, loop back
    4. Repeats until a final answer or max_iterations is reached

    Usage:
        loop = AgentLoop(llm=llm, tools=[calc_tool], memory=memory)
        answer = loop.run("What is 25 * 37?")
        print(answer)  # "925"
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[Any] | None = None,
        memory: BaseMemory | None = None,
        max_iterations: int = 10,
        system_prompt: str = "You are a helpful AI assistant.",
        verbose: bool = True,
    ):
        self.llm = llm
        self.max_iterations = max_iterations
        self.logger = AgentLogger(verbose=verbose)

        # Set up memory
        from ..memory.buffer import BufferMemory

        self.memory = memory or BufferMemory(max_messages=50)

        # Set up tool registry and executor
        self.registry = ToolRegistry()
        if tools:
            self.registry.register_many(tools)
        self.executor = ToolExecutor(self.registry, self.logger)

        # Add system prompt to memory
        self.memory.add_message("system", system_prompt)

        self.logger.system(
            f"AgentLoop initialized",
            f"Model: {llm}\n"
            f"Tools: {self.registry.tool_names}\n"
            f"Max iterations: {max_iterations}",
        )

    def run(self, user_input: str) -> str:
        """
        Run the agent loop with a user query.

        This is the main entry point. It:
        1. Adds the user message to memory
        2. Enters the loop
        3. Returns the final answer

        Args:
            user_input: The user's question or request.

        Returns:
            The agent's final text response.
        """
        self.logger.separator()
        self.logger.system(f"User: {user_input}")

        # Add user message to memory
        self.memory.add_message("user", user_input)

        # Get tool schemas (empty list if no tools registered)
        tool_schemas = self.registry.to_openai_schema()

        # ──────────────────────────────────────────────────────────
        # THE LOOP — this is where the magic happens
        # ──────────────────────────────────────────────────────────
        for iteration in range(1, self.max_iterations + 1):
            self.logger.system(f"Iteration {iteration}/{self.max_iterations}")

            # Step 1: Get current messages from memory
            messages = self.memory.get_messages()

            # Step 2: Call the LLM (with or without tools)
            if tool_schemas:
                response = self.llm.chat_with_tools(messages, tool_schemas)
            else:
                response = self.llm.chat(messages)

            # Step 3: Check if the LLM wants to call tools
            if response.has_tool_calls:
                self.logger.thought("LLM requested tool calls")

                # Record the assistant's tool-call message in memory
                # This is CRITICAL — the API requires the assistant message
                # with tool_calls to appear before the tool result messages
                self.memory.add_assistant_tool_calls(
                    content=response.content,
                    tool_calls=response.tool_calls,
                )

                # Step 4: Execute each tool call
                results = self.executor.execute_many(response.tool_calls)

                # Step 5: Add tool results to memory
                for result in results:
                    self.memory.add_tool_message(
                        tool_call_id=result["tool_call_id"],
                        content=result["content"],
                    )

                # Loop back to step 1 — the LLM will see the tool results
                continue

            # Step 6: No tool calls → this is the final answer!
            final_answer = response.content
            self.memory.add_message("assistant", final_answer)
            self.logger.result(final_answer)
            return final_answer

        # If we hit max iterations, return whatever we have
        self.logger.error(
            f"Max iterations ({self.max_iterations}) reached without final answer"
        )
        return (
            f"I was unable to complete the task within {self.max_iterations} "
            f"iterations. Here's what I know so far based on my analysis."
        )

    def reset(self) -> None:
        """Clear memory and start fresh."""
        self.memory.clear()
        self.logger.system("Memory cleared")
