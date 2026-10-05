"""
Example 9: Agent Registry — one agent, many independent processes.

Run these as SEPARATE commands (separate processes, even on different days):

    python examples/09_agent_registry.py intake     # process A creates + uses the agent
    python examples/09_agent_registry.py followup   # process B continues from A's exact state
    python examples/09_agent_registry.py billing    # restricted process: redacted view, still writes back
    python examples/09_agent_registry.py history    # versions, provenance, audit

Every session commits an immutable, hash-addressed version. If two processes
commit at the same time, the second is three-way merged (or forked/rejected).
"""

import sys

from lightagentx import AccessPolicy, AgentRegistry, OpenAILLM

REGISTRY = AgentRegistry("~/.lightx/agents-demo")


def get_agent() -> str:
    try:
        return REGISTRY.find("SupportBot")
    except KeyError:
        policy = AccessPolicy(owner="alice", private_keys={"card_number"})
        return REGISTRY.create(
            "SupportBot", owner="alice", policy=policy,
            system_prompt="You are a customer-support agent. Keep answers short.",
            schema={"customer": "str", "decisions": "list", "card_number": "str"},
        )


def main(step: str) -> None:
    aid = get_agent()
    llm = OpenAILLM()

    if step == "intake":
        with REGISTRY.attach(aid, "alice", llm=llm, workflow="intake", state_tools=True) as s:
            print(s.run("Customer ACME (card 4111 1111 1111 1111) wants a refund for order 17. "
                        "Record the customer and the decision in your state."))
            s.state["card_number"] = "4111 1111 1111 1111"

    elif step == "followup":
        with REGISTRY.attach(aid, "alice", llm=llm, workflow="followup") as s:
            print("Restored state:", s.state)
            print(s.run("What did we decide for this customer, and why?"))

    elif step == "billing":
        REGISTRY.grant(aid, "alice", "billing", {"write"})
        with REGISTRY.attach(aid, "billing", llm=llm, workflow="billing") as s:
            print("Billing sees:", s.state)  # no card_number
            print(s.run("Confirm the refund was issued."))
            s.state.setdefault("decisions", []).append("refund issued")

    elif step == "history":
        for v in REGISTRY.log(aid, "alice"):
            p = v.provenance
            print(f"{v.short}  parents={[x[:8] for x in v.parents]}  {p['principal']:<8} "
                  f"{p['workflow']:<9} pid={p['pid']}  {v.message}")
        print("verified versions:", REGISTRY.verify(aid))
        for e in REGISTRY.audit_log(aid, "alice")[-6:]:
            print(f"  {e['decision']:<8} {e['principal']:<8} {e['action']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "intake")
