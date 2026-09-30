"""Lifecycle Hooks — event-driven middleware for agent observability and control."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class HookEvent:
    """
    Data object passed to every hook callback.

    Attributes:
        name: The event name (e.g. ``"before_llm_call"``).
        data: Arbitrary payload — keys depend on the event type.
        timestamp: When the event was emitted (monotonic seconds).
    """

    name: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)


_KNOWN_EVENTS = frozenset({
    "before_llm_call",
    "after_llm_call",
    "before_tool_call",
    "after_tool_call",
    "on_iteration",
    "on_error",
    "on_agent_start",
    "on_agent_end",
})


class HookRegistry:
    """
    A registry for lifecycle hook callbacks.

    Register callbacks for agent lifecycle events and they will be called
    automatically at the appropriate points during execution.

    Supported events::

        before_llm_call   — data: messages, model
        after_llm_call    — data: messages, model, response
        before_tool_call  — data: tool_name, arguments
        after_tool_call   — data: tool_name, arguments, result
        on_iteration      — data: iteration, max_iterations
        on_error          — data: error, context
        on_agent_start    — data: agent_name, input_text
        on_agent_end      — data: agent_name, input_text, output

    Usage::

        hooks = HookRegistry()

        @hooks.on("before_llm_call")
        def log_it(event):
            print(f"Calling {event.data['model']}")

        agent = SingleAgent(name="Bot", llm=llm, hooks=hooks)
    """

    def __init__(self) -> None:
        self._hooks: dict[str, list[Callable[[HookEvent], None]]] = {}

    def on(self, event_name: str) -> Callable:
        """
        Decorator to register a callback for an event.

        Args:
            event_name: One of the supported lifecycle event names.

        Returns:
            The original function (unmodified), so it can still be called
            directly if needed.
        """
        def decorator(func: Callable[[HookEvent], None]) -> Callable:
            self.register(event_name, func)
            return func
        return decorator

    def register(self, event_name: str, callback: Callable[[HookEvent], None]) -> None:
        """Programmatically register a callback for an event."""
        self._hooks.setdefault(event_name, []).append(callback)

    def emit(self, event_name: str, **data: Any) -> None:
        """
        Fire an event. All registered callbacks are called in order.

        Hook errors are silently caught so they never crash the agent.
        """
        if event_name not in self._hooks:
            return

        event = HookEvent(name=event_name, data=data)
        for callback in self._hooks[event_name]:
            try:
                callback(event)
            except Exception:
                pass  # hooks must never crash the agent

    def clear(self, event_name: str | None = None) -> None:
        """
        Remove hooks.

        Args:
            event_name: If given, clear only hooks for that event.
                        If ``None``, clear all hooks.
        """
        if event_name is None:
            self._hooks.clear()
        else:
            self._hooks.pop(event_name, None)

    @property
    def registered_events(self) -> list[str]:
        """List event names that have at least one hook registered."""
        return [k for k, v in self._hooks.items() if v]

    def __repr__(self) -> str:
        counts = {k: len(v) for k, v in self._hooks.items() if v}
        return f"HookRegistry({counts})"
