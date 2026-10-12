import { motion } from "motion/react";
import { Archive, FileCheck2, Lock, Upload } from "lucide-react";
import { CodeBlock, Reveal, Section } from "./ui";
import { DOCS, ease } from "../lib";

/* ── version graph ─────────────────────────────────────────── */

const NODES = [
  { id: "a1f3", x: 40, y: 70, lane: "main", label: "create" },
  { id: "7c0e", x: 140, y: 70, lane: "main", label: "session · alice" },
  { id: "e92b", x: 250, y: 70, lane: "main", label: "session · proc A" },
  { id: "3d4a", x: 250, y: 150, lane: "side", label: "session · proc B" },
  { id: "b58f", x: 370, y: 70, lane: "main", label: "3-way merge" },
  { id: "0c71", x: 470, y: 70, lane: "main", label: "session · bob" },
];

const EDGES = [
  "M40 70 L140 70",
  "M140 70 L250 70",
  "M140 70 C190 70 200 150 250 150",
  "M250 70 L370 70",
  "M250 150 C310 150 320 70 370 70",
  "M370 70 L470 70",
];

function VersionGraph() {
  return (
    <svg viewBox="0 0 520 200" className="h-auto w-full" role="img" aria-label="Two processes commit to the same agent at once; their versions are merged into main">
      {EDGES.map((d, i) => (
        <motion.path
          key={d}
          d={d}
          fill="none"
          stroke={i === 2 || i === 4 ? "#38bdf8" : "#22c55e"}
          strokeOpacity={0.7}
          strokeWidth={2}
          initial={{ pathLength: 0 }}
          whileInView={{ pathLength: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.3 + i * 0.25, ease }}
        />
      ))}
      {NODES.map((n, i) => (
        <motion.g
          key={n.id}
          initial={{ opacity: 0, scale: 0.4 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ type: "spring", stiffness: 400, damping: 22, delay: 0.2 + i * 0.25 }}
          style={{ transformOrigin: `${n.x}px ${n.y}px` }}
        >
          <circle cx={n.x} cy={n.y} r={n.label.includes("merge") ? 9 : 7} fill="#0f172a" stroke={n.lane === "side" ? "#38bdf8" : "#22c55e"} strokeWidth={2.5} />
          <text x={n.x} y={n.lane === "side" ? n.y + 28 : n.y - 22} textAnchor="middle" className="fill-fg font-mono text-[11px]">
            {n.id}…
          </text>
          <text x={n.x} y={n.lane === "side" ? n.y + 43 : n.y - 38} textAnchor="middle" className="fill-faint text-[10px]">
            {n.label}
          </text>
        </motion.g>
      ))}
    </svg>
  );
}

/* ── concurrency result (evals/results/REPORT.md, table 1a) ── */

const LOSS = [
  { procs: 2, lightx: 0, langgraph: 42 },
  { procs: 4, lightx: 0, langgraph: 52 },
  { procs: 8, lightx: 0, langgraph: 72 },
  { procs: 12, lightx: 0, langgraph: 74 },
];

function LossChart() {
  return (
    <figure>
      <figcaption className="mb-5">
        <div className="font-medium">Writes silently lost under concurrency</div>
        <div className="mt-1 text-sm text-muted">N processes write to one agent at the same instant. Lower is better.</div>
      </figcaption>
      <div className="mb-4 flex flex-wrap gap-4 text-xs text-muted">
        <span className="inline-flex items-center gap-1.5"><span className="size-2.5 rounded-sm bg-accent" /> AgentRegistry (merge)</span>
        <span className="inline-flex items-center gap-1.5"><span className="size-2.5 rounded-sm bg-rose" /> LangGraph + SQLite checkpointer</span>
      </div>
      <div className="space-y-4">
        {LOSS.map((r, i) => (
          <div key={r.procs} className="grid grid-cols-[4.5rem_1fr] items-center gap-3">
            <span className="text-right font-mono text-xs text-faint">{r.procs} procs</span>
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <span className="h-2.5 w-1 rounded-sm bg-accent" />
                <span className="font-mono text-xs text-accent">0%</span>
              </div>
              <div className="flex items-center gap-2">
                <motion.div
                  className="h-2.5 rounded-sm bg-rose"
                  initial={{ width: 0 }}
                  whileInView={{ width: `${r.langgraph * 0.85}%` }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.9, delay: i * 0.12, ease }}
                />
                <span className="font-mono text-xs text-rose">{r.langgraph}%</span>
              </div>
            </div>
          </div>
        ))}
      </div>
      <p className="mt-5 text-xs leading-relaxed text-faint">
        20 ops per process, 3 repetitions. LangGraph's lost events still exist in orphaned checkpoints, but the next reader
        of the agent doesn't see them.{" "}
        <a className="underline decoration-line-strong underline-offset-2 hover:text-fg" href="https://github.com/devdaveddev/LightAgentX/blob/main/evals/results/REPORT.md">
          Full report and method
        </a>
      </p>
    </figure>
  );
}

/* ── .lxagent shipping ─────────────────────────────────────── */

const SHIP = [
  { icon: Archive, title: "export_agent()", body: "History, config and tool source go into one .lxagent file. With redact=True it exports only what a read principal may see." },
  { icon: Upload, title: "SupportBot.lxagent", body: "A gzip tar with a manifest. Each version is hashed by its own content, so the archive can't be quietly edited." },
  { icon: FileCheck2, title: "import_agent()", body: "It rejects unsafe paths, unlisted files and any version that fails its hash, then restores branches and metadata." },
  { icon: Lock, title: "load_tools(trust=True)", body: "Nothing in the archive runs on import. tool_files() lists the shipped code so you can review it first." },
];

const PERMS = ["read", "run", "write", "fork", "merge", "private", "admin"];

export function Registry() {
  return (
    <Section
      id="registry"
      eyebrow="Agent Registry"
      title="Agents that persist, with git-like history."
      lead="Any process can find an agent, attach to it and continue its work. Every session commits an immutable, SHA-256 addressed version. When two processes write at once, the registry merges, forks or rejects. It never silently overwrites."
    >
      <div className="grid gap-6 lg:grid-cols-2">
        <Reveal className="flex flex-col gap-6">
          <CodeBlock
            title="example_09_agent_registry.py"
            code={`from lightagentx import AgentRegistry

reg = AgentRegistry("~/.lightx/agents")
aid = reg.create("SupportBot", owner="alice",
                 schema={"customer": "str", "decisions": "list"})

with reg.attach(aid, "alice", llm=llm) as s:   # any process, any time
    s.run("Customer ACME wants a refund")
    s.state["customer"] = "ACME"                 # committed as a new version`}
          />
          <div className="rounded-2xl border border-line bg-surface p-5">
            <VersionGraph />
            <div className="mt-2 flex flex-wrap gap-2">
              {["log", "diff", "time travel", "verify()", "on_conflict = merge · fork · reject"].map((t) => (
                <span key={t} className="rounded-md border border-line bg-surface-2 px-2 py-1 font-mono text-[11px] text-muted">
                  {t}
                </span>
              ))}
            </div>
          </div>
        </Reveal>
        <Reveal delay={0.1} className="flex flex-col gap-6">
          <div className="rounded-2xl border border-line bg-surface p-6">
            <LossChart />
          </div>
          <div className="rounded-2xl border border-line bg-surface p-6">
            <div className="font-medium">Permissions for each principal</div>
            <p className="mt-1 text-sm text-muted">
              Writers can be scoped to branches, state keys can be private, and secrets are redacted. A restricted session runs on a redacted copy and still writes back without losing data.
            </p>
            <div className="mt-4 flex flex-wrap gap-2">
              {PERMS.map((p, i) => (
                <motion.span
                  key={p}
                  initial={{ opacity: 0, y: 6 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.05, duration: 0.3 }}
                  className="rounded-full border border-accent/30 bg-accent/[0.06] px-3 py-1 font-mono text-xs text-accent"
                >
                  {p}
                </motion.span>
              ))}
            </div>
          </div>
        </Reveal>
      </div>

      <Reveal className="mt-16">
        <h3 className="text-xl font-semibold tracking-tight">Move an agent to another machine as a single file.</h3>
        <p className="mt-2 max-w-2xl text-muted">
          Export writes the whole agent to one <code className="font-mono text-fg">.lxagent</code> archive, and import checks every hash before restoring it.
        </p>
      </Reveal>
      <div className="relative mt-8">
        <motion.div
          className="absolute top-6 right-[12%] left-[12%] hidden h-px origin-left bg-gradient-to-r from-accent via-sky to-accent md:block"
          initial={{ scaleX: 0 }}
          whileInView={{ scaleX: 1 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 1.2, ease }}
          aria-hidden
        />
        <ol className="relative grid gap-4 md:grid-cols-4">
        {SHIP.map((s, i) => (
          <motion.li
            key={s.title}
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-80px" }}
            transition={{ duration: 0.6, ease, delay: 0.15 + i * 0.15 }}
            className="relative h-full rounded-2xl border border-line bg-surface p-5"
          >
            <span className="relative z-10 inline-flex size-12 items-center justify-center rounded-xl border border-line-strong bg-surface-2 text-accent">
              <s.icon className="size-5" />
            </span>
            <div className="mt-4 font-mono text-sm text-fg">{s.title}</div>
            <p className="mt-2 text-sm leading-relaxed text-muted">{s.body}</p>
          </motion.li>
        ))}
        </ol>
      </div>
      <Reveal className="mt-6">
        <a href={`${DOCS}/agent-registry.md`} className="text-sm text-muted underline decoration-line-strong underline-offset-4 hover:text-fg">
          Design, guarantees and limitations →
        </a>
      </Reveal>
    </Section>
  );
}
