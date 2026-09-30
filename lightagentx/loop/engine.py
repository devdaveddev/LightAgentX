"""Agent Loop Engine — core reasoning and execution loop."""

from __future__ import annotations

from typing import Any

from ..llm.base import BaseLLM
from ..memory.base import BaseMemory
from ..tools.registry import ToolRegistry
from ..tools.executor import ToolExecutor
from ..utils.logger import AgentLogger
from ..hooks import HookRegistry


class AgentLoop:
    """
    The core reasoning loop that drives tool-calling agents.

    This loop:
    1. Sends the conversation (with tool schemas) to the LLM
    2. If the LLM returns a text response → finished
    3. If the LLM requests tool calls → execute them, add results, repeat
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
        hooks: HookRegistry | None = None,
    ):
        self.llm = llm
        self.max_iterations = max_iterations
        self.logger = AgentLogger(verbose=verbose)
        self.hooks = hooks or HookRegistry()

        from ..memory.buffer import BufferMemory

        self.memory = memory if memory is not None else BufferMemory(max_messages=50)

        self.registry = ToolRegistry()
        if tools:
            self.registry.register_many(tools)
        self.executor = ToolExecutor(self.registry, self.logger)

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

        Args:
            user_input: The user's question or request.

        Returns:
            The agent's final text response.
        """
        self.logger.separator()
        self.logger.system(f"User: {user_input}")

        self.memory.add_message("user", user_input)

        tool_schemas = self.registry.to_openai_schema()

        for iteration in range(1, self.max_iterations + 1):
            self.logger.system(f"Iteration {iteration}/{self.max_iterations}")
            self.hooks.emit(
                "on_iteration",
                iteration=iteration,
                max_iterations=self.max_iterations,
            )

            messages = self.memory.get_messages()

            self.hooks.emit(
                "before_llm_call",
                messages=messages,
                model=self.llm.model,
            )

            try:
                if tool_schemas:
                    response = self.llm.chat_with_tools(messages, tool_schemas)
                else:
                    response = self.llm.chat(messages)
            except Exception as exc:
                self.hooks.emit("on_error", error=exc, context="llm_call")
                raise

            self.hooks.emit(
                "after_llm_call",
                messages=messages,
                model=self.llm.model,
                response=response,
            )

            if response.has_tool_calls:
                self.logger.thought("LLM requested tool calls")

                self.memory.add_assistant_tool_calls(
                    content=response.content,
                    tool_calls=response.tool_calls,
                )

                for tc in response.tool_calls:
                    self.hooks.emit(
                        "before_tool_call",
                        tool_name=tc["name"],
                        arguments=tc.get("arguments", {}),
                    )

                results = self.executor.execute_many(response.tool_calls)

                for i, result in enumerate(results):
                    self.hooks.emit(
                        "after_tool_call",
                        tool_name=response.tool_calls[i]["name"],
                        arguments=response.tool_calls[i].get("arguments", {}),
                        result=result["content"],
                    )
                    self.memory.add_tool_message(
                        tool_call_id=result["tool_call_id"],
                        content=result["content"],
                    )

                continue

            final_answer = response.content
            self.memory.add_message("assistant", final_answer)
            self.logger.result(final_answer)
            return final_answer

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
