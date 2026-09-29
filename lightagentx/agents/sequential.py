"""
Sequential Pipeline — chains multiple agents so output flows from one to the next.

HOW THIS MAPS TO LANGCHAIN:
  - LangChain's `SequentialChain` / LCEL `chain1 | chain2 | chain3`
  - CrewAI's sequential process: Agent A → Agent B → Agent C

WHAT YOU LEARN HERE:
  This is the simplest multi-agent pattern:
  - Agent 1 processes the input and produces output
  - Agent 2 takes Agent 1's output as input
  - Agent 3 takes Agent 2's output... and so on

  Example use case: Researcher → Writer → Editor pipeline
"""

from __future__ import annotations

from .base import BaseAgent
from ..utils.logger import AgentLogger


class SequentialPipeline(BaseAgent):
    """
    Chain multiple agents sequentially.

    Each agent's output becomes the next agent's input.

    Usage:
        researcher = SingleAgent(name="Researcher", ...)
        writer = SingleAgent(name="Writer", ...)
        editor = SingleAgent(name="Editor", ...)

        pipeline = SequentialPipeline(
            name="Content Pipeline",
            agents=[researcher, writer, editor],
        )
        result = pipeline.run("Write about quantum computing")
    """

    def __init__(
        self,
        name: str,
        agents: list[BaseAgent],
        description: str = "",
        verbose: bool = True,
    ):
        super().__init__(name=name, description=description)
        self.agents = agents
        self.logger = AgentLogger(verbose=verbose)

    def run(self, input_text: str) -> str:
        """
        Run agents sequentially, passing output to the next.

        THE FLOW:
            input → Agent1 → output1 → Agent2 → output2 → ... → final output

        Each intermediate output is formatted so the next agent understands
        the context it's receiving.
        """
        self.logger.separator()
        self.logger.system(
            f"Sequential Pipeline: {self.name}",
            f"Agents: {[a.name for a in self.agents]}",
        )

        current_input = input_text

        for i, agent in enumerate(self.agents, 1):
            self.logger.separator()
            self.logger.agent(
                agent.name,
                f"Step {i}/{len(self.agents)} — processing...",
            )

            # Format input for downstream agents to include context
            if i > 1:
                formatted_input = (
                    f"You are step {i} in a pipeline. "
                    f"Here is the output from the previous step:\n\n"
                    f"---\n{current_input}\n---\n\n"
                    f"Original task: {input_text}\n\n"
                    f"Your role ({agent.name}): {agent.description}\n"
                    f"Please process the above and produce your output."
                )
            else:
                formatted_input = current_input

            # Run the agent
            current_input = agent.run(formatted_input)

            self.logger.agent(
                agent.name,
                f"Step {i} complete — output length: {len(current_input)} chars",
            )

        self.logger.separator()
        self.logger.result(f"Pipeline '{self.name}' complete")

        return current_input
