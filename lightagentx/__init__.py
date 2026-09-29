"""
LightAgentX — A lightweight modular agentic AI framework.

Built from scratch to understand how LangChain and similar frameworks work.

Quick Start:
    from lightagentx import OpenAILLM, SingleAgent, tool

    llm = OpenAILLM()

    @tool
    def calculate(expression: str) -> str:
        '''Evaluate a math expression.'''
        return str(eval(expression))

    agent = SingleAgent(name="MathBot", llm=llm, tools=[calculate])
    print(agent.run("What is 25 * 37?"))
"""

__version__ = "0.1.0"

# ── LLM ──────────────────────────────────────────────────────────
from .llm.base import BaseLLM, LLMResponse
from .llm.openai_llm import OpenAILLM

# ── Memory ───────────────────────────────────────────────────────
from .memory.base import BaseMemory
from .memory.buffer import BufferMemory
from .memory.summary import SummaryMemory

# ── Tools ────────────────────────────────────────────────────────
from .tools.base import BaseTool, tool
from .tools.registry import ToolRegistry
from .tools.executor import ToolExecutor

# ── Planning ─────────────────────────────────────────────────────
from .planning.base import BasePlanner, Step
from .planning.react import ReActPlanner

# ── Loop Engine ──────────────────────────────────────────────────
from .loop.engine import AgentLoop

# ── Agents ───────────────────────────────────────────────────────
from .agents.base import BaseAgent
from .agents.single import SingleAgent
from .agents.sequential import SequentialPipeline
from .agents.crew import CrewAgent

# ── Utilities ────────────────────────────────────────────────────
from .utils.logger import AgentLogger

__all__ = [
    # LLM
    "BaseLLM", "LLMResponse", "OpenAILLM",
    # Memory
    "BaseMemory", "BufferMemory", "SummaryMemory",
    # Tools
    "BaseTool", "tool", "ToolRegistry", "ToolExecutor",
    # Planning
    "BasePlanner", "Step", "ReActPlanner",
    # Loop
    "AgentLoop",
    # Agents
    "BaseAgent", "SingleAgent", "SequentialPipeline", "CrewAgent",
    # Utils
    "AgentLogger",
]
