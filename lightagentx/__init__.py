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

from .llm.base import BaseLLM, LLMResponse
from .llm.key_guard import SecureKey
from .llm.openai_llm import OpenAILLM
from .llm.router import SmartRouter

try:
    from .llm.anthropic_llm import AnthropicLLM
except ImportError:
    pass

try:
    from .llm.gemini_llm import GeminiLLM
except ImportError:
    pass

from .memory.base import BaseMemory
from .memory.buffer import BufferMemory
from .memory.summary import SummaryMemory

from .tools.base import BaseTool, tool
from .tools.registry import ToolRegistry
from .tools.executor import ToolExecutor

from .planning.base import BasePlanner, Step
from .planning.react import ReActPlanner

from .loop.engine import AgentLoop

from .agents.base import BaseAgent
from .agents.single import SingleAgent
from .agents.sequential import SequentialPipeline
from .agents.crew import CrewAgent

from .hooks import HookRegistry, HookEvent
from .snapshot import AgentSnapshot

from .sandbox import Risk, Sandbox, SandboxPolicy, SandboxViolation
from .state import AccessPolicy, AgentRegistry
from .features import (
    SmartOSDisabledError, disable_smartos, enable_smartos, smartos_enabled, smartos_status,
)

from .utils.logger import AgentLogger

__all__ = [
    # LLM
    "BaseLLM", "LLMResponse", "SecureKey",
    "OpenAILLM", "AnthropicLLM", "GeminiLLM", "SmartRouter",
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
    # Hooks & Snapshots
    "HookRegistry", "HookEvent", "AgentSnapshot",
    # Sandbox (SmartOS lives in lightagentx.smartos — needs the [os] extra)
    "Risk", "Sandbox", "SandboxPolicy", "SandboxViolation",
    # Persistent, versioned, permissioned agents
    "AgentRegistry", "AccessPolicy",
    # SmartOS on/off switch
    "smartos_enabled", "smartos_status", "enable_smartos", "disable_smartos",
    "SmartOSDisabledError",
    # Utils
    "AgentLogger",
]
