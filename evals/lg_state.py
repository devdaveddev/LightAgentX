"""LangGraph baseline state (module level so LangGraph can resolve its type hints)."""

import operator
from typing import Annotated, TypedDict


class EventsState(TypedDict):
    events: Annotated[list, operator.add]
