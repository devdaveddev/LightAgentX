"""Exporting and importing agents (.lxagent archives)."""

import io
import json
import tarfile
from pathlib import Path

import pytest

from lightagentx.llm.base import BaseLLM, LLMResponse
from lightagentx.state import (
    AccessDenied,
    AccessPolicy,
    AgentRegistry,
    ArchiveError,
    IntegrityError,
)

TOOL_SOURCE = '''
from lightagentx import tool

MARKER = []

@tool(risk="low")
def order_status(order_id: str) -> str:
    """Look up an order.

    Args:
        order_id: Order number.
    """
    return f"{order_id}: shipped"
'''


class Scripted(BaseLLM):
    def __init__(self, replies):
        super().__init__(model="scripted")
        self.replies, self.seen = list(replies), []

    def chat(self, messages):
        self.seen.append(messages)
        r = self.replies.pop(0) if self.replies else "ok"
        return r if isinstance(r, LLMResponse) else LLMResponse(content=r)

    def chat_with_tools(self, messages, tools):
        return self.chat(messages)


@pytest.fixture
def src(tmp_path):
    reg = AgentRegistry(tmp_path / "machine-A")
    policy = AccessPolicy(owner="priya", private_keys={"card_number"})
    aid = reg.create("SupportBot", owner="priya", system_prompt="Be brief.", policy=policy,
                     schema={"customer": "str", "card_number": "str"},
                     state={"card_number": "4111 1111 1111 1111"})
    with reg.attach(aid, "priya", llm=Scripted(["Noted, refund approved."])) as s:
        s.run("ACME wants a refund. Their api_key=sk-lxeval-ABCDEFGHIJKLMNOPQRSTU")
        s.state["customer"] = "ACME"
    reg.fork(aid, "priya", "experiment")
    with reg.attach(aid, "priya", branch="experiment") as s:
        s.state["customer"] = "ACME (trial)"
    tool_file = tmp_path / "support_tools.py"
    tool_file.write_text(TOOL_SOURCE)
    return reg, aid, tool_file, tmp_path


def rebuild(archive: Path, out: Path, edit) -> Path:
    """Copy an archive, letting `edit(members)` change it."""
    with tarfile.open(archive) as tar:
        members = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    edit(members)
    with tarfile.open(out, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return out


class TestRoundTrip:
    def test_full_history_moves_unchanged(self, src):
        reg, aid, _, tmp = src
        archive = reg.export_agent(aid, "priya", tmp / "SupportBot.lxagent")
        dst = AgentRegistry(tmp / "machine-B")
        assert dst.import_agent(archive, "devansh") == aid
        assert dst.branches(aid, "priya") == reg.branches(aid, "priya")
        assert [v.id for v in dst.log(aid, "priya")] == [v.id for v in reg.log(aid, "priya")]
        assert dst.read(aid, "priya") == reg.read(aid, "priya")
        assert dst.verify(aid) == reg.verify(aid)

    def test_imported_agent_continues_the_conversation(self, src):
        reg, aid, _, tmp = src
        dst = AgentRegistry(tmp / "machine-B")
        dst.import_agent(reg.export_agent(aid, "priya", tmp / "a.lxagent"), "devansh")
        llm = Scripted(["We approved a refund for ACME."])
        with dst.attach(aid, "priya", llm=llm) as s:
            s.run("What did we decide?")
        seen = [m["content"] for m in llm.seen[0]]
        assert "Noted, refund approved." in seen          # the original conversation travelled
        assert len(dst.log(aid, "priya")) == len(reg.log(aid, "priya")) + 1

    def test_selected_branches_only(self, src):
        reg, aid, _, tmp = src
        archive = reg.export_agent(aid, "priya", tmp / "a.lxagent", branches=["main"])
        dst = AgentRegistry(tmp / "machine-B")
        dst.import_agent(archive, "devansh")
        assert list(dst.branches(aid, "priya")) == ["main"]

    def test_import_as_copy_rekeys_history(self, src):
        reg, aid, _, tmp = src
        archive = reg.export_agent(aid, "priya", tmp / "a.lxagent")
        copy_id = reg.import_agent(archive, "priya", agent_id="agt_supportcopy", owner="devansh")
        assert reg.verify(copy_id) == reg.verify(aid)
        assert set(reg.branches(copy_id, "devansh").values()).isdisjoint(reg.branches(aid, "priya").values())
        assert reg.read(copy_id, "devansh")["state"] == reg.read(aid, "priya")["state"]
        assert reg._meta(copy_id)["owner"] == "devansh"
        with pytest.raises(AccessDenied):
            reg.read(copy_id, "priya")                    # ownership really moved

    def test_import_refuses_to_overwrite(self, src):
        reg, aid, _, tmp = src
        archive = reg.export_agent(aid, "priya", tmp / "a.lxagent")
        with pytest.raises(ValueError, match="already exists"):
            reg.import_agent(archive, "priya")

    def test_export_and_import_are_audited(self, src):
        reg, aid, _, tmp = src
        archive = reg.export_agent(aid, "priya", tmp / "a.lxagent")
        dst = AgentRegistry(tmp / "machine-B")
        dst.import_agent(archive, "devansh")
        assert any(e["action"] == "export" for e in reg.audit_log(aid, "priya"))
        assert any(e["action"] == "import" and e["principal"] == "devansh"
                   for e in dst.audit_log(aid, "priya"))


class TestPermissions:
    def test_full_export_needs_admin(self, src):
        reg, aid, _, tmp = src
        reg.grant(aid, "priya", "billing", {"read"})
        with pytest.raises(AccessDenied):
            reg.export_agent(aid, "billing", tmp / "a.lxagent")

    def test_redacted_export_hides_private_data_and_history(self, src):
        reg, aid, _, tmp = src
        reg.grant(aid, "priya", "billing", {"read"})
        archive = reg.export_agent(aid, "billing", tmp / "a.lxagent", redact=True)
        with tarfile.open(archive) as tar:
            blob = b"".join(tar.extractfile(m).read() for m in tar.getmembers())
        assert b"4111" not in blob and b"sk-lxeval-ABCDEF" not in blob
        dst = AgentRegistry(tmp / "machine-B")
        dst.import_agent(archive, "billing", owner="billing")
        assert "card_number" not in dst.read(aid, "billing")["state"]
        assert all(len(v.parents) == 0 for v in dst.log(aid, "billing", ref="main"))


class TestIntegrity:
    def test_modified_file_is_rejected(self, src):
        reg, aid, _, tmp = src
        archive = reg.export_agent(aid, "priya", tmp / "a.lxagent")

        def tamper(members):
            name = next(n for n in members if n.startswith("objects/") and b"ACME" in members[n])
            members[name] = members[name].replace(b"ACME", b"EVIL")

        bad = rebuild(archive, tmp / "bad.lxagent", tamper)
        with pytest.raises(IntegrityError):
            AgentRegistry(tmp / "machine-B").import_agent(bad, "devansh")

    def test_recomputed_manifest_still_fails_version_hash(self, src):
        """An attacker who also fixes the manifest still can't change a version."""
        reg, aid, _, tmp = src
        archive = reg.export_agent(aid, "priya", tmp / "a.lxagent")

        def tamper(members):
            import hashlib
            name = next(n for n in members if n.startswith("objects/") and b"ACME" in members[n])
            members[name] = members[name].replace(b"ACME", b"EVIL")
            manifest = json.loads(members["manifest.json"])
            manifest["files"][name] = hashlib.sha256(members[name]).hexdigest()
            members["manifest.json"] = json.dumps(manifest).encode()

        bad = rebuild(archive, tmp / "bad.lxagent", tamper)
        with pytest.raises(IntegrityError, match="hash"):
            AgentRegistry(tmp / "machine-B").import_agent(bad, "devansh")

    def test_unlisted_file_is_rejected(self, src):
        reg, aid, _, tmp = src
        archive = reg.export_agent(aid, "priya", tmp / "a.lxagent")
        bad = rebuild(archive, tmp / "bad.lxagent",
                      lambda m: m.__setitem__("tools/evil.py", b"import os"))
        with pytest.raises(IntegrityError, match="manifest"):
            AgentRegistry(tmp / "machine-B").import_agent(bad, "devansh")

    def test_path_traversal_is_rejected(self, src):
        reg, aid, _, tmp = src
        archive = reg.export_agent(aid, "priya", tmp / "a.lxagent")
        bad = rebuild(archive, tmp / "bad.lxagent",
                      lambda m: m.__setitem__("../../escape.txt", b"x"))
        with pytest.raises(ArchiveError, match="Unsafe"):
            AgentRegistry(tmp / "machine-B").import_agent(bad, "devansh")
        assert not (tmp.parent / "escape.txt").exists()

    def test_not_an_archive(self, tmp_path):
        f = tmp_path / "x.lxagent"
        f.write_text("hello")
        with pytest.raises(ArchiveError):
            AgentRegistry(tmp_path / "r").import_agent(f, "me")


class TestTools:
    def test_tools_travel_and_need_explicit_trust(self, src):
        reg, aid, tool_file, tmp = src
        archive = reg.export_agent(aid, "priya", tmp / "a.lxagent", tools=[tool_file])
        dst = AgentRegistry(tmp / "machine-B")
        dst.import_agent(archive, "devansh")
        assert list(dst.tool_files(aid)) == ["support_tools.py"]
        with pytest.raises(PermissionError, match="trust=True"):
            dst.load_tools(aid, "priya")
        tools = dst.load_tools(aid, "priya", trust=True)
        assert [t.name for t in tools] == ["order_status"]
        assert tools[0](order_id="ORD-1") == "ORD-1: shipped"

    def test_tool_objects_are_resolved_to_their_source_file(self, src, monkeypatch):
        reg, aid, tool_file, tmp = src
        import importlib.util
        spec = importlib.util.spec_from_file_location("support_tools_mod", tool_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        archive = reg.export_agent(aid, "priya", tmp / "a.lxagent", tools=[mod.order_status])
        with tarfile.open(archive) as tar:
            assert "tools/support_tools.py" in tar.getnames()

    def test_import_never_executes_tool_code(self, src):
        reg, aid, _, tmp = src
        boom = tmp / "boom.py"
        canary = tmp / "EXECUTED"
        boom.write_text(f"open({str(canary)!r}, 'w').write('x')\n")
        archive = reg.export_agent(aid, "priya", tmp / "a.lxagent", tools=[boom])
        AgentRegistry(tmp / "machine-B").import_agent(archive, "devansh")
        assert not canary.exists()

    def test_warns_when_tool_files_miss_expected_tools(self, src):
        reg, aid, tool_file, tmp = src
        with reg.attach(aid, "priya", llm=Scripted(["x"]), tools=[]) as s:
            pass
        # make the agent "expect" a tool the shipped file doesn't define
        empty = tmp / "empty_tools.py"
        empty.write_text("x = 1\n")
        from lightagentx.state.store import Version
        head = reg.store.get(reg.branches(aid, "priya")["main"])
        v = Version.create(agent_id=aid, parents=(head.id,), created_at=head.created_at + 1,
                           message="m", provenance=head.provenance, config=head.config,
                           transcript=head.transcript, state=head.state, tool_manifest=("refund",))
        reg.store.put(v)
        reg.store.cas_ref(aid, "main", head.id, v.id)
        with pytest.warns(UserWarning, match="refund"):
            reg.export_agent(aid, "priya", tmp / "a.lxagent", tools=[empty])
