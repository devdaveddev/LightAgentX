"""Worker run in separate OS processes by tests/test_registry.py."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightagentx.llm.base import BaseLLM, LLMResponse  # noqa: E402
from lightagentx.state import AgentRegistry, ConflictError  # noqa: E402


class EchoLLM(BaseLLM):
    """Answers with how many messages it was given, so tests can see prior context."""

    def __init__(self, tag: str):
        super().__init__(model="echo")
        self.tag = tag
        self.seen: list[list[dict]] = []

    def chat(self, messages):
        self.seen.append(messages)
        return LLMResponse(content=f"{self.tag}: saw {len(messages)} messages")

    def chat_with_tools(self, messages, tools):
        return self.chat(messages)


def main() -> None:
    cmd, root = sys.argv[1], sys.argv[2]
    reg = AgentRegistry(root)

    if cmd == "create_and_chat":  # process A
        aid = reg.create("Support", owner="alice", system_prompt="You help customers.",
                         schema={"customer": "str", "decisions": "list"})
        with reg.attach(aid, "alice", llm=EchoLLM("A"), workflow="intake") as s:
            s.run("Customer ABC wants a refund for order 17")
            s.state["customer"] = "ABC"
            s.state["decisions"] = ["refund approved"]
        print(json.dumps({"agent_id": aid, "pid": os.getpid()}))

    elif cmd == "continue":  # process B — knows only the agent's name
        aid = reg.find("Support")
        llm = EchoLLM("B")
        with reg.attach(aid, "alice", llm=llm, workflow="followup") as s:
            restored_state = dict(s.state)
            answer = s.run("What did we decide for this customer?")
        first_call = llm.seen[0]
        print(json.dumps({
            "pid": os.getpid(), "state": restored_state, "answer": answer,
            "llm_saw": [m["content"] for m in first_call],
        }))

    elif cmd == "append_events":  # concurrency hammer
        aid, worker, n, mode = sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
        conflicts = 0
        for i in range(n):
            try:
                with reg.attach(aid, "alice", on_conflict=mode) as s:
                    s.state.setdefault("events", []).append(f"{worker}-{i}")
            except ConflictError:
                conflicts += 1
        print(json.dumps({"worker": worker, "conflicts": conflicts}))


if __name__ == "__main__":
    main()
