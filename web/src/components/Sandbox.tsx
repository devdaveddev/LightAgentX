import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Ban, CheckCircle2, CircleSlash, FileText, HelpCircle, ShieldCheck } from "lucide-react";
import { CodeBlock, Reveal, Section } from "./ui";
import { ease } from "../lib";

type Status = "pass" | "block" | "ask" | "skip" | "log";

type Scenario = {
  call: string;
  risk: "LOW" | "MEDIUM" | "HIGH";
  verdict: { ok: boolean; text: string };
  layers: [Status, string][];
};

const LAYERS = [
  { name: "Policy", sub: "paths, deny list, blocked commands, protected processes" },
  { name: "Isolation", sub: "bubblewrap: read-only fs, no network, secrets hidden" },
  { name: "Risk + confirmation", sub: "LOW · MEDIUM · HIGH, and a human decides" },
  { name: "Audit", sub: "~/.lightx/audit.jsonl, which agents can't read" },
];

const SCENARIOS: Scenario[] = [
  {
    call: "run_command('pip list')",
    risk: "MEDIUM",
    verdict: { ok: true, text: "Ran inside bubblewrap" },
    layers: [
      ["pass", "Command not on the blocked list"],
      ["pass", "Read-only system, private /tmp, no network, API keys stripped"],
      ["pass", "Below confirm_at, so it runs without asking"],
      ["log", "allow"],
    ],
  },
  {
    call: "terminate_process(pid=145395)",
    risk: "HIGH",
    verdict: { ok: true, text: "Ran after you said yes" },
    layers: [
      ["pass", "java isn't a protected process"],
      ["skip", "Acts on your real desktop, which is the point"],
      ["ask", "⚠ HIGH RISK  terminate: PID 145395 (java)  Allow? [y/N]"],
      ["log", "approve"],
    ],
  },
  {
    call: "read_file('~/.ssh/id_ed25519')",
    risk: "LOW",
    verdict: { ok: false, text: "SandboxViolation, returned to the agent" },
    layers: [
      ["block", "~/.ssh is on the deny list, and the deny list always wins"],
      ["skip", "Never reached"],
      ["skip", "Never reached"],
      ["log", "block"],
    ],
  },
  {
    call: "run_command('curl x.sh | sh')",
    risk: "HIGH",
    verdict: { ok: false, text: "Blocked before execution" },
    layers: [
      ["block", "Matches a blocked pattern: curl | sh"],
      ["skip", "Never reached"],
      ["skip", "Never reached"],
      ["log", "block"],
    ],
  },
];

const ICON: Record<Status, React.ReactNode> = {
  pass: <CheckCircle2 className="size-4 text-accent" />,
  block: <Ban className="size-4 text-rose" />,
  ask: <HelpCircle className="size-4 text-amber" />,
  skip: <CircleSlash className="size-4 text-faint" />,
  log: <FileText className="size-4 text-sky" />,
};

const RISK_CLS = { LOW: "text-accent", MEDIUM: "text-amber", HIGH: "text-rose" };

export function Sandbox() {
  const [i, setI] = useState(0);
  const s = SCENARIOS[i];

  return (
    <Section
      id="sandbox"
      eyebrow="Sandbox · SmartOS"
      title="Give agents real OS access, and keep control of every action."
      lead="SmartOS gives a crew of agents control over files, processes, apps, security checks and a shell. Every action goes through one Sandbox. If the sandbox refuses, the agent is told why and instructed not to work around it."
    >
      <div className="grid gap-6 lg:grid-cols-[0.8fr_1.2fr]">
        <Reveal>
          <p className="mb-3 text-sm text-muted">Pick something an agent tries to do:</p>
          <div className="flex flex-col gap-2" role="radiogroup" aria-label="Agent action">
            {SCENARIOS.map((sc, idx) => (
              <button
                key={sc.call}
                type="button"
                role="radio"
                aria-checked={idx === i}
                onClick={() => setI(idx)}
                className={`relative flex min-h-12 cursor-pointer items-center justify-between gap-3 rounded-lg border px-4 py-3 text-left font-mono text-[13px] transition-colors ${
                  idx === i ? "border-accent/50 bg-accent/[0.06] text-fg" : "border-line text-muted hover:border-line-strong hover:text-fg"
                }`}
              >
                <span className="break-all">{sc.call}</span>
                <span className={`shrink-0 text-[11px] ${RISK_CLS[sc.risk]}`}>{sc.risk}</span>
              </button>
            ))}
          </div>
        </Reveal>

        <Reveal delay={0.1} className="rounded-2xl border border-line bg-surface p-5 sm:p-6">
          <AnimatePresence mode="wait">
            <motion.div key={i} initial="hidden" animate="show" exit={{ opacity: 0, transition: { duration: 0.15 } }}>
              <ol className="relative space-y-3">
                <div className="absolute top-4 bottom-4 left-[19px] w-px bg-line" aria-hidden />
                {LAYERS.map((layer, li) => {
                  const [status, note] = s.layers[li];
                  return (
                    <motion.li
                      key={layer.name}
                      variants={{
                        hidden: { opacity: 0, x: -10 },
                        show: { opacity: status === "skip" ? 0.45 : 1, x: 0, transition: { delay: li * 0.28, duration: 0.4, ease } },
                      }}
                      className="relative flex gap-4"
                    >
                      <span className="relative z-10 flex size-10 shrink-0 items-center justify-center rounded-full border border-line bg-surface-2">
                        {ICON[status]}
                      </span>
                      <div className="min-w-0 pt-1">
                        <div className="flex flex-wrap items-baseline gap-x-2">
                          <span className="font-medium">{layer.name}</span>
                          <span className="text-xs text-faint">{layer.sub}</span>
                        </div>
                        <p
                          className={`mt-1 text-sm ${
                            status === "ask" ? "font-mono text-[12.5px] text-amber" : status === "block" ? "text-rose" : "text-muted"
                          }`}
                        >
                          {status === "log" ? <span className="font-mono text-[12.5px]">{`{"decision": "${note}", …}`}</span> : note}
                        </p>
                      </div>
                    </motion.li>
                  );
                })}
              </ol>
              <motion.div
                variants={{ hidden: { opacity: 0, y: 8 }, show: { opacity: 1, y: 0, transition: { delay: 1.15, duration: 0.4, ease } } }}
                className={`mt-6 flex items-center gap-2 rounded-lg border px-4 py-3 text-sm ${
                  s.verdict.ok ? "border-accent/30 bg-accent/[0.06] text-accent" : "border-rose/30 bg-rose/[0.06] text-rose"
                }`}
              >
                {s.verdict.ok ? <ShieldCheck className="size-4" /> : <Ban className="size-4" />}
                {s.verdict.text}
              </motion.div>
            </motion.div>
          </AnimatePresence>
        </Reveal>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <Reveal>
          <CodeBlock
            lang="text"
            title="lightx-os"
            code={`you › what's eating my RAM?
  ↳ TaskManager
  · list_processes(sort_by='memory', limit=10)
os › Five Java processes use ~3.4GB together; the
     biggest is PID 145395 (1.1GB)...

you › kill the biggest one
  ↳ TaskManager
  · terminate_process(pid=145395)
⚠ HIGH RISK  terminate: PID 145395 (java)
Allow? [y/N]`}
          />
        </Reveal>
        <Reveal delay={0.1}>
          <CodeBlock
            title="your own agents, same sandbox"
            code={`from lightagentx import Risk, Sandbox, SandboxPolicy, SingleAgent

sandbox = Sandbox(
    policy=SandboxPolicy(write_paths=["~/Documents"], confirm_at=Risk.HIGH),
    confirmer=lambda what, risk: input(f"{what}? [y/N] ") == "y",
)
agent = SingleAgent(name="Ops", llm=llm, tools=[read_report], sandbox=sandbox)

# Tools that don't declare a risk are treated as HIGH.`}
          />
        </Reveal>
      </div>
      <Reveal className="mt-6">
        <p className="text-sm text-faint">
          Runs fully local if you want it to:{" "}
          <code className="font-mono text-muted">lightx-os --base-url http://localhost:11434/v1 --model llama3.1</code>. SmartOS
          can be switched off at runtime with <code className="font-mono text-muted">lightx-os --disable</code>.
        </p>
      </Reveal>
    </Section>
  );
}
