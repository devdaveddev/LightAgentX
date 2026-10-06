"""Tests for the AgentRegistry: persistence, discovery, versioning, concurrency, permissions."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lightagentx import SingleAgent
from lightagentx.llm.base import BaseLLM, LLMResponse
from lightagentx.state import (
    AccessDenied,
    AccessPolicy,
    AgentRegistry,
    ConflictError,
    IntegrityError,
    MergeConflictError,
    merge_lists,
    merge_state,
)
from lightagentx.tools.base import tool

WORKER = Path(__file__).with_name("registry_worker.py")


def worker(*args, check=True):
    r = subprocess.run([sys.executable, str(WORKER), *map(str, args)],
                       capture_output=True, text=True, timeout=120)
    if check and r.returncode != 0:
        raise AssertionError(r.stderr)
    return json.loads(r.stdout.strip().splitlines()[-1])


class Scripted(BaseLLM):
    def __init__(self, replies=()):
        super().__init__(model="scripted")
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages):
        self.seen.append(messages)
        r = self.replies.pop(0) if self.replies else "ok"
        return r if isinstance(r, LLMResponse) else LLMResponse(content=r)

    chat_with_tools = lambda self, m, t: self.chat(m)


@tool
def lookup(order: int) -> str:
    """Look up an order.

    Args:
        order: Order number.
    """
    return f"order {order}: card ending 4242, api_key=sk-live{'x' * 24}"


@pytest.fixture
def reg(tmp_path):
    return AgentRegistry(tmp_path / "agents")


def make_agent(reg, **kw):
    return reg.create("Support", owner="alice", system_prompt="You help customers.",
                      schema={"customer": "str", "decisions": "list", "ssn": "str"}, **kw)


def assert_valid_transcript(messages):
    """Every tool call is answered before the next user turn (providers require this)."""
    pending = set()
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            assert not pending
            pending = {tc["id"] for tc in m["tool_calls"]}
        elif m["role"] == "tool":
            pending.discard(m["tool_call_id"])
        else:
            assert not pending, f"unanswered tool calls before {m['role']}"
    assert not pending


# ── The five questions, across real OS processes ─────────────────────────

class TestAcrossProcesses:
    def test_agent_outlives_process_and_is_continued_by_another(self, tmp_path):
        root = tmp_path / "agents"
        a = worker("create_and_chat", root)          # process A creates, chats, exits
        b = worker("continue", root)                  # process B: finds by name, continues

        assert a["pid"] != b["pid"]
        assert b["state"] == {"customer": "ABC", "decisions": ["refund approved"]}
        # B's LLM received A's actual conversation, not a retrieval of it
        assert b["llm_saw"][:4] == [
            "You help customers.",
            "Customer ABC wants a refund for order 17",
            "A: saw 2 messages",
            "What did we decide for this customer?",
        ]
        assert b["answer"] == "B: saw 4 messages"

        reg = AgentRegistry(root)
        log = reg.log(a["agent_id"], "alice")
        assert [v.provenance["workflow"] for v in log[:2]] == ["followup", "intake"]
        assert {v.provenance["pid"] for v in log[:2]} == {a["pid"], b["pid"]}

    @pytest.mark.parametrize("mode", ["merge"])
    def test_no_lost_updates_under_concurrent_processes(self, tmp_path, mode):
        root = tmp_path / "agents"
        reg = AgentRegistry(root)
        aid = reg.create("Counter", owner="alice", schema={"events": "list"})
        procs = [subprocess.Popen([sys.executable, str(WORKER), "append_events", str(root), aid,
                                   f"w{i}", "20", mode], stdout=subprocess.PIPE, text=True)
                 for i in range(12)]
        results = [json.loads(p.communicate(timeout=300)[0]) for p in procs]
        assert all(p.returncode == 0 for p in procs)
        assert sum(r["conflicts"] for r in results) == 0  # merge mode never gives up
        events = reg.read(aid, "alice")["state"]["events"]
        assert sorted(events) == sorted(f"w{i}-{j}" for i in range(12) for j in range(20))
        assert reg.verify(aid) >= 241  # every version intact and reachable

    def test_reject_mode_surfaces_conflicts_instead_of_merging(self, tmp_path):
        root = tmp_path / "agents"
        reg = AgentRegistry(root)
        aid = reg.create("Counter", owner="alice", schema={"events": "list"})
        procs = [subprocess.Popen([sys.executable, str(WORKER), "append_events", str(root), aid,
                                   f"w{i}", "5", "reject"], stdout=subprocess.PIPE, text=True)
                 for i in range(6)]
        results = [json.loads(p.communicate(timeout=120)[0]) for p in procs]
        conflicts = sum(r["conflicts"] for r in results)
        events = reg.read(aid, "alice")["state"].get("events", [])
        assert len(events) + conflicts == 30  # every attempt either landed or was rejected loudly


# ── Versioning, fork, merge ──────────────────────────────────────────────

class TestVersioning:
    def test_every_commit_is_an_immutable_content_addressed_version(self, reg):
        aid = make_agent(reg)
        with reg.attach(aid, "alice") as s:
            s.state["customer"] = "ABC"
        v1 = reg.read(aid, "alice")["version"]
        with reg.attach(aid, "alice") as s:
            s.state["customer"] = "XYZ"
        assert reg.read(aid, "alice", v1[:10])["state"]["customer"] == "ABC"  # time travel
        assert reg.diff(aid, "alice", v1, "main")["state"] == {"customer": ("ABC", "XYZ")}

    def test_no_change_no_version(self, reg):
        aid = make_agent(reg)
        with reg.attach(aid, "alice"):
            pass
        assert len(reg.log(aid, "alice")) == 1

    def test_failed_session_is_discarded(self, reg):
        aid = make_agent(reg)
        with pytest.raises(RuntimeError):
            with reg.attach(aid, "alice") as s:
                s.state["customer"] = "half-done"
                raise RuntimeError("workflow crashed")
        assert "customer" not in reg.read(aid, "alice")["state"]

    def test_tampering_is_detected(self, reg):
        aid = make_agent(reg)
        with reg.attach(aid, "alice") as s:
            s.state["customer"] = "ABC"
        vid = reg.read(aid, "alice")["version"]
        obj = reg.store._obj_path(vid)
        obj.write_text(obj.read_text().replace("ABC", "EVIL"))
        with pytest.raises(IntegrityError):
            reg.verify(aid)

    def test_concurrent_sessions_auto_merge_transcripts_and_state(self, reg):
        aid = make_agent(reg)
        s1 = reg.attach(aid, "alice", llm=Scripted(["refund ok"]))
        s2 = reg.attach(aid, "alice", llm=Scripted(["address updated"]))
        s1.run("refund order 17"); s1.state["decisions"] = ["refund"]
        s2.run("change address"); s2.state["decisions"] = ["new address"]
        s1.commit()
        merged = s2.commit()  # branch moved under s2 -> three-way merge
        assert len(merged.parents) == 2
        st = reg.read(aid, "alice")
        assert sorted(st["state"]["decisions"]) == ["new address", "refund"]
        contents = [m["content"] for m in st["transcript"]]
        assert any("Parallel session" in c and "refund order 17" in c for c in contents)
        assert "change address" in contents
        assert_valid_transcript(st["transcript"])

    def test_merge_keeps_tool_call_pairs_intact(self, reg):
        aid = make_agent(reg)
        tc = lambda i: LLMResponse(tool_calls=[{"id": f"c{i}", "name": "lookup", "arguments": {"order": i}}])
        s1 = reg.attach(aid, "alice", llm=Scripted([tc(1), "done 1"]), tools=[lookup])
        s2 = reg.attach(aid, "alice", llm=Scripted([tc(2), "done 2"]), tools=[lookup])
        s1.run("check 1"); s2.run("check 2")
        s1.commit(); s2.commit()
        assert_valid_transcript(reg.read(aid, "alice")["transcript"])

    def test_structured_conflict_raises_then_resolves(self, reg):
        aid = make_agent(reg)
        for strategy, expected in [("raise", None), ("theirs", "FIRST"), ("ours", "SECOND")]:
            with reg.attach(aid, "alice") as s:
                s.state["customer"] = "BASE"
            a = reg.attach(aid, "alice")
            b = reg.attach(aid, "alice", merge_strategy=strategy)
            a.state["customer"] = "FIRST"; a.commit()
            b.state["customer"] = "SECOND"
            if expected is None:
                with pytest.raises(MergeConflictError):
                    b.commit()
            else:
                b.commit()
                st = reg.read(aid, "alice")["state"]
                assert st["customer"] == expected
                assert "customer" in st["_merge_notes"][-1]["resolved_conflicts"][0]

    def test_fork_mode_puts_conflicting_work_on_a_new_branch(self, reg):
        aid = make_agent(reg)
        a = reg.attach(aid, "alice")
        b = reg.attach(aid, "alice", on_conflict="fork")
        a.state["customer"] = "A"; a.commit()
        b.state["customer"] = "B"; b.commit()
        branches = reg.branches(aid, "alice")
        fork = next(n for n in branches if n.startswith("main.fork-"))
        assert reg.read(aid, "alice", fork)["state"]["customer"] == "B"
        assert reg.read(aid, "alice")["state"]["customer"] == "A"

    def test_explicit_fork_and_merge(self, reg):
        aid = make_agent(reg)
        reg.fork(aid, "alice", "experiment")
        with reg.attach(aid, "alice", branch="experiment") as s:
            s.state["decisions"] = ["try discount"]
        ff = reg.merge(aid, "alice", "experiment")            # fast-forward
        assert reg.branches(aid, "alice")["main"] == ff.id
        with reg.attach(aid, "alice", branch="experiment") as s:
            s.state["decisions"] = ["try discount", "escalate"]
        with reg.attach(aid, "alice") as s:
            s.state["customer"] = "ABC"
        m = reg.merge(aid, "alice", "experiment")               # true merge
        st = reg.read(aid, "alice")["state"]
        assert st["customer"] == "ABC" and st["decisions"] == ["try discount", "escalate"]
        assert m.parents[0] == reg.log(aid, "alice")[1].id or len(m.parents) == 2

    def test_time_travel_session_is_read_only(self, reg):
        aid = make_agent(reg)
        v0 = reg.read(aid, "alice")["version"]
        with reg.attach(aid, "alice") as s:
            s.state["customer"] = "ABC"
        old = reg.attach(aid, "alice", at=v0[:12])
        assert "customer" not in old.state
        old.state["customer"] = "rewrite history"
        with pytest.raises(AccessDenied):
            old.commit()

    def test_digest_merge_uses_summarizer(self, reg):
        aid = make_agent(reg)
        a = reg.attach(aid, "alice", llm=Scripted(["A done"]))
        b = reg.attach(aid, "alice", llm=Scripted(["B done"]), transcript_strategy="digest",
                       summarizer=lambda msgs: f"{len(msgs)} msgs about refunds")
        a.run("refund"); a.commit()
        b.run("address"); b.commit()
        st = reg.read(aid, "alice")
        assert st["state"]["_merge_notes"][-1]["digest"] == "2 msgs about refunds"
        assert all("refund" != m["content"] for m in st["transcript"])
        with reg.attach(aid, "alice", llm=Scripted(["x"])) as s:
            assert "2 msgs about refunds" in s.agent.system_prompt


# ── Permissions ──────────────────────────────────────────────────────────

class TestPermissions:
    def setup_agent(self, reg):
        aid = make_agent(reg, state={"customer": "ABC", "ssn": "123-45-6789"})
        reg.set_private_keys(aid, "alice", {"ssn"})
        with reg.attach(aid, "alice", llm=Scripted([
            LLMResponse(tool_calls=[{"id": "c1", "name": "lookup", "arguments": {"order": 17}}]),
            "looked up"]), tools=[lookup]) as s:
            s.run("look up order 17")
        return aid

    def test_strangers_see_nothing(self, reg):
        aid = self.setup_agent(reg)
        assert reg.list_agents("mallory") == []
        for call in (lambda: reg.read(aid, "mallory"), lambda: reg.attach(aid, "mallory")):
            with pytest.raises(AccessDenied):
                call()
        denied = [e for e in reg.audit_log(aid, "alice") if e["decision"] == "denied"]
        assert {e["principal"] for e in denied} == {"mallory"}

    def test_redacted_view_hides_private_keys_and_secrets(self, reg):
        aid = self.setup_agent(reg)
        reg.grant(aid, "alice", "billing", {"read"})
        view = reg.read(aid, "billing")
        assert "ssn" not in view["state"]
        text = json.dumps(view["transcript"])
        assert "sk-live" not in text and "[REDACTED]" in text
        full = json.dumps(reg.read(aid, "alice")["transcript"])
        assert "sk-live" in full

    def test_redacted_session_writes_back_without_losing_hidden_data(self, reg):
        aid = self.setup_agent(reg)
        reg.grant(aid, "alice", "billing", {"write"})
        llm = Scripted(["noted"])
        with reg.attach(aid, "billing", llm=llm, tools=[lookup]) as s:
            assert "ssn" not in s.state
            assert "sk-live" not in json.dumps(llm.seen) and "sk-live" not in json.dumps(s.transcript)
            s.run("mark invoice paid")
            s.state["decisions"] = ["invoice paid"]
        full = reg.read(aid, "alice")
        assert full["state"]["ssn"] == "123-45-6789"                 # private key untouched
        assert "sk-live" in json.dumps(full["transcript"])            # original kept unredacted
        assert full["transcript"][-1]["content"] == "noted"           # new turns appended
        assert full["state"]["decisions"] == ["invoice paid"]

    def test_cannot_write_private_key_without_right(self, reg):
        aid = self.setup_agent(reg)
        reg.grant(aid, "alice", "billing", {"write"})
        s = reg.attach(aid, "billing")
        s.state["ssn"] = "000"
        with pytest.raises(AccessDenied):
            s.commit()

    def test_run_only_principal_cannot_change_the_agent(self, reg):
        aid = self.setup_agent(reg)
        reg.grant(aid, "alice", "kiosk", {"run"})
        before = reg.read(aid, "alice")["version"]
        with reg.attach(aid, "kiosk", llm=Scripted(["hi"]), tools=[lookup]) as s:
            s.run("hello")
        assert reg.read(aid, "alice")["version"] == before

    def test_branch_restricted_writer_lands_on_personal_branch(self, reg):
        aid = self.setup_agent(reg)
        reg.grant(aid, "alice", "intern", {"write", "fork"}, write_branches=["drafts-*"])
        with reg.attach(aid, "intern") as s:
            s.state["decisions"] = ["intern idea"]
        assert "decisions" not in reg.read(aid, "alice")["state"]
        assert reg.read(aid, "alice", "intern.main")["state"]["decisions"] == ["intern idea"]

    def test_only_admin_can_grant(self, reg):
        aid = self.setup_agent(reg)
        reg.grant(aid, "alice", "billing", {"write"})
        with pytest.raises(AccessDenied):
            reg.grant(aid, "billing", "billing", {"admin"})
        reg.revoke(aid, "alice", "billing")
        with pytest.raises(AccessDenied):
            reg.read(aid, "billing")

    def test_merge_needs_merge_right(self, reg):
        aid = self.setup_agent(reg)
        reg.grant(aid, "alice", "bob", {"write", "fork"})
        reg.fork(aid, "bob", "bob-branch")
        with pytest.raises(AccessDenied):
            reg.merge(aid, "bob", "bob-branch")


# ── Misc ─────────────────────────────────────────────────────────────────

class TestMisc:
    def test_state_tools_let_the_llm_update_structured_state(self, reg):
        aid = make_agent(reg)
        llm = Scripted([LLMResponse(tool_calls=[{"id": "c1", "name": "set_state",
                                                 "arguments": {"key": "customer", "value_json": "\"ABC\""}}]),
                        "saved"])
        with reg.attach(aid, "alice", llm=llm, state_tools=True) as s:
            s.run("remember the customer is ABC")
        assert reg.read(aid, "alice")["state"]["customer"] == "ABC"

    def test_schema_is_enforced(self, reg):
        aid = make_agent(reg)
        s = reg.attach(aid, "alice")
        s.state["customer"] = 42
        with pytest.raises(TypeError):
            s.commit()

    def test_register_existing_agent(self, reg):
        agent = SingleAgent(name="Legacy", llm=Scripted(["hi there"]), verbose=False)
        agent.run("hello")
        aid = reg.register(agent, owner="alice")
        with reg.attach(aid, "alice", llm=Scripted(["back"])) as s:
            assert [m["content"] for m in s.transcript] == ["hello", "hi there"]

    def test_ref_names_cannot_escape(self, reg):
        aid = make_agent(reg)
        with pytest.raises(KeyError):
            reg.read(aid, "alice", "../meta.json")

    def test_merge_helpers(self):
        assert merge_lists([1, 2, 3], [1, 3, 4], [1, 2, 3, 5]) == [1, 3, 4, 5]
        merged, _ = merge_state({"a": {"x": 1}}, {"a": {"x": 1, "y": 2}}, {"a": {"x": 1, "z": 3}})
        assert merged == {"a": {"x": 1, "y": 2, "z": 3}}
        with pytest.raises(MergeConflictError):
            merge_state({"a": 1}, {"a": 2}, {"a": 3})
        assert merge_state({"a": 1}, {"a": 2}, {"a": 3}, lambda p, b, o, t: o + t)[0] == {"a": 5}
