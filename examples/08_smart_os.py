"""
Example 8: SmartOS — agents that manage your computer from a sandbox.

Every OS action (files, processes, apps, commands) goes through one Sandbox:
  - path rules (secrets like ~/.ssh are never readable)
  - command isolation (bubblewrap on Linux: read-only system, no network)
  - human confirmation for HIGH-risk actions (kill, delete, overwrite)
  - an audit log of every decision

For the interactive terminal, just run:  lightx-os
"""

from lightagentx import OpenAILLM, Risk, Sandbox, SandboxPolicy
from lightagentx.smartos import SmartOS


def ask_human(description: str, risk: Risk) -> bool:
    return input(f"[{risk.name}] {description} — allow? [y/N] ").strip().lower() == "y"


def main():
    sandbox = Sandbox(
        policy=SandboxPolicy(
            allow_network=False,          # sandboxed commands get no network
            confirm_at=Risk.HIGH,         # ask before kill / delete / overwrite
            trusted_apps=["firefox"],     # launch without asking
        ),
        confirmer=ask_human,
    )
    print(sandbox.describe())

    smart_os = SmartOS(llm=OpenAILLM(), sandbox=sandbox, mode="crew")

    for request in [
        "How is my system doing? Anything using lots of memory?",
        "Run a security scan and tell me if anything needs attention.",
        "Create a file called plan.txt in the workspace listing three things to clean up.",
    ]:
        print(f"\n>>> {request}")
        print(smart_os.run(request))

    print("\nAudit log:")
    for entry in sandbox.audit_log:
        print(f"  {entry.decision:<9} {entry.risk:<6} {entry.action:<12} {entry.target[:60]}")


if __name__ == "__main__":
    main()
