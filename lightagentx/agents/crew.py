"""Crew Agent — role-based agent collaboration with a manager."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseAgent
from ..llm.base import BaseLLM
from ..utils.logger import AgentLogger


_MANAGER_SYSTEM_PROMPT = """You are a team manager coordinating specialist agents to solve a task.

You have the following specialist agents available:
{agent_descriptions}

Your job:
1. Analyze the task
2. Decide which agent(s) to delegate to
3. Specify what each agent should do

Respond with a JSON array of delegations:
[
    {{"agent": "agent_name", "task": "specific task for this agent"}},
    ...
]

Rules:
- Use agent names EXACTLY as listed above
- Each delegation must have "agent" and "task" keys
- You may delegate to one or more agents
- Be specific about what each agent should do
- Respond ONLY with the JSON array, no other text
"""

_SYNTHESIS_PROMPT = """You are synthesizing results from multiple specialist agents into a final coherent answer.

Original task: {original_task}

Agent results:
{agent_results}

Instructions:
- Combine the insights from all agents
- Produce a single, coherent, well-structured response
- Credit agents' contributions where appropriate
- Resolve any contradictions between agent outputs
"""


class CrewAgent(BaseAgent):
    """
    A crew of specialist agents managed by a coordinator LLM.

    The manager analyzes the task, delegates subtasks to specialists,
    and synthesizes their results into a final answer.

    Usage:
        researcher = SingleAgent(name="Researcher", description="Expert at finding information", ...)
        analyst = SingleAgent(name="Analyst", description="Expert at data analysis", ...)

        crew = CrewAgent(
            name="Research Team",
            agents=[researcher, analyst],
            manager_llm=OpenAILLM(),
        )
        result = crew.run("Analyze the impact of AI on healthcare")
    """

    def __init__(
        self,
        name: str,
        agents: list[BaseAgent],
        manager_llm: BaseLLM,
        description: str = "",
        verbose: bool = True,
    ):
        super().__init__(name=name, description=description)
        self.agents = {agent.name: agent for agent in agents}
        self.manager_llm = manager_llm
        self.logger = AgentLogger(verbose=verbose)

    def run(self, input_text: str) -> str:
        """Run the crew on a task."""
        self.logger.separator()
        self.logger.system(
            f"Crew: {self.name}",
            f"Agents: {list(self.agents.keys())}",
        )

        delegations = self._get_delegations(input_text)
        self.logger.plan(
            "Manager created delegation plan:",
            json.dumps(delegations, indent=2),
        )

        results: list[dict[str, str]] = []
        for delegation in delegations:
            agent_name = delegation.get("agent", "")
            task = delegation.get("task", "")

            agent = self.agents.get(agent_name)
            if agent is None:
                self.logger.error(
                    f"Agent '{agent_name}' not found, skipping. "
                    f"Available: {list(self.agents.keys())}"
                )
                results.append(
                    {
                        "agent": agent_name,
                        "result": f"Error: Agent '{agent_name}' not found",
                    }
                )
                continue

            self.logger.separator()
            self.logger.agent(agent_name, f"Assigned: {task[:100]}")

            result = agent.run(task)
            results.append({"agent": agent_name, "result": result})

            self.logger.agent(agent_name, f"Done — {len(result)} chars")

        self.logger.separator()
        self.logger.system("Manager synthesizing results...")
        final = self._synthesize(input_text, results)

        self.logger.result(f"Crew '{self.name}' complete")
        return final

    def _get_delegations(self, task: str) -> list[dict[str, str]]:
        """Ask the manager LLM to create a delegation plan."""
        agent_desc = "\n".join(
            f"- {name}: {agent.description}"
            for name, agent in self.agents.items()
        )

        system_prompt = _MANAGER_SYSTEM_PROMPT.format(
            agent_descriptions=agent_desc
        )

        response = self.manager_llm.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Task: {task}"},
            ]
        )

        try:
            content = response.content.strip()
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            delegations = json.loads(content)
            if isinstance(delegations, list):
                return delegations
        except (json.JSONDecodeError, IndexError):
            self.logger.error(
                "Failed to parse manager's delegation plan, "
                "falling back to broadcasting task to all agents"
            )

        return [
            {"agent": name, "task": task} for name in self.agents
        ]

    def _synthesize(
        self, original_task: str, results: list[dict[str, str]]
    ) -> str:
        """Ask the manager LLM to synthesize agent results."""
        formatted_results = "\n\n".join(
            f"### {r['agent']}\n{r['result']}" for r in results
        )

        prompt = _SYNTHESIS_PROMPT.format(
            original_task=original_task,
            agent_results=formatted_results,
        )

        response = self.manager_llm.chat(
            [{"role": "user", "content": prompt}]
        )

        return response.content
