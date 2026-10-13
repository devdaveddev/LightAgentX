import { useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "motion/react";
import { ChevronDown } from "lucide-react";
import { CodeBlock, Section } from "./ui";
import { DOCS, REPO, ease } from "../lib";

const C = ({ children }: { children: ReactNode }) => (
  <code className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[0.85em] text-fg">{children}</code>
);

const STEPS: { title: string; body: ReactNode; code?: string; lang?: "bash" | "python" }[] = [
  {
    title: "1. Install Python 3.10 or newer",
    body: <>Check with <C>python3 --version</C>. Use a virtual environment so LightAgentX's packages don't mix with your system's.</>,
    code: `python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate`,
    lang: "bash",
  },
  {
    title: "2. Install LightAgentX",
    body: (
      <>
        The base install needs only <C>openai</C>. Add extras for other providers or for SmartOS:{" "}
        <C>[anthropic]</C>, <C>[gemini]</C>, <C>[os]</C> (adds <C>psutil</C>), or <C>[all]</C>.
      </>
    ),
    code: `pip install lightagentx
pip install "lightagentx[all]"   # everything`,
    lang: "bash",
  },
  {
    title: "3. Choose a model",
    body: (
      <>
        Set one API key. The <C>lightx</C> commands pick a provider automatically: a <C>--base-url</C> first, then{" "}
        <C>ANTHROPIC_API_KEY</C>, <C>OPENAI_API_KEY</C>, <C>GOOGLE_API_KEY</C>. For no key at all, run a local model with
        Ollama.
      </>
    ),
    code: `export OPENAI_API_KEY=sk-...        # or ANTHROPIC_API_KEY / GOOGLE_API_KEY

# free and local instead:
ollama pull llama3.1
lightx-os --base-url http://localhost:11434/v1 --model llama3.1`,
    lang: "bash",
  },
  {
    title: "4. Run your first agent",
    body: <>Clone the repo for the examples, or paste the tool-agent snippet from the quickstart above.</>,
    code: `git clone ${REPO}.git && cd LightAgentX
python examples/02_tool_agent.py`,
    lang: "bash",
  },
  {
    title: "5. Optional: real isolation for SmartOS and doctor (Linux)",
    body: (
      <>
        Shell commands run under bubblewrap when it's installed and working. Without it LightAgentX falls back to plain
        subprocesses, which are not isolated, so every command needs your approval.
      </>
    ),
    code: `sudo dnf install bubblewrap      # Fedora
sudo apt install bubblewrap      # Debian / Ubuntu
bwrap --ro-bind / / --unshare-all true && echo "isolation works"`,
    lang: "bash",
  },
  {
    title: "6. Where things are stored",
    body: (
      <>
        Everything lives under <C>~/.lightx/</C>: <C>agents/</C> (registry), <C>workspace/</C> (the only folder sandboxed
        commands can write), <C>trash/</C> (recoverable deletes), <C>audit.jsonl</C>, <C>config.json</C> (SmartOS on/off)
        and <C>doctor/backups/</C>.
      </>
    ),
  },
];

const ISSUES: { q: string; a: ReactNode; code?: string }[] = [
  {
    q: "OpenAI API key not found (or Anthropic / Google)",
    a: <>The client couldn't find a key. Set the matching environment variable in the same shell, or pass <C>api_key=</C>. A local server still needs some value: <C>OpenAILLM(base_url=..., api_key="local")</C>.</>,
    code: `export OPENAI_API_KEY=sk-...
echo $OPENAI_API_KEY        # check it's set in this terminal`,
  },
  {
    q: "No LLM configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY or GOOGLE_API_KEY…",
    a: <>Shown by <C>lightx-os</C> and <C>lightx doctor</C> when no key is set and no <C>--base-url</C> was given. Set a key, or point at a local server.</>,
    code: `lightx doctor --base-url http://localhost:11434/v1 --model qwen2.5:3b`,
  },
  {
    q: "The 'anthropic' (or 'google-genai') package is required…",
    a: <>Those providers are optional extras. Install the one you use.</>,
    code: `pip install "lightagentx[anthropic]"    # or [gemini], or [all]`,
  },
  {
    q: "ModuleNotFoundError: psutil, or SmartOS tools fail to start",
    a: <>SmartOS needs the <C>[os]</C> extra.</>,
    code: `pip install "lightagentx[os]"`,
  },
  {
    q: "SmartOSDisabledError, or lightx-os only explains how to re-enable",
    a: <>SmartOS was switched off, either in <C>~/.lightx/config.json</C> or with <C>LIGHTAGENTX_SMARTOS=0</C> (the environment variable wins).</>,
    code: `lightx-os --status
lightx-os --enable
unset LIGHTAGENTX_SMARTOS`,
  },
  {
    q: "Every shell command asks for approval / \"not isolated\"",
    a: <>bubblewrap isn't installed or can't create namespaces, so the unisolated fallback is in use. Install it (setup step 5). Some containers and hardened kernels disable unprivileged user namespaces; then the fallback is expected.</>,
    code: `bwrap --ro-bind / / --unshare-all true; echo $?   # 0 means isolation works`,
  },
  {
    q: "The agent says: Error: … SandboxViolation: <reason>",
    a: <>Working as intended: the policy refused the action (deny list, blocked command, protected process or a path outside what's allowed). To allow more, widen the policy deliberately.</>,
    code: `lightx-os --allow-write ~/Documents     # extra writable path
lightx-os --network                      # give sandboxed commands network access`,
  },
  {
    q: "Max iterations (10) reached.",
    a: <>The model kept calling tools without giving a final answer. Usually the task is too vague or a tool keeps returning errors; check the verbose trace. Raise the cap if the task really needs more steps.</>,
    code: `agent = SingleAgent(name="Bot", llm=llm, tools=tools, verbose=True)   # see each step`,
  },
  {
    q: "A small local model ignores tools or answers badly",
    a: <>Small models do better with fewer tools at once. Use SmartOS in single-agent mode, or try a larger model such as <C>qwen2.5:7b</C> (needs about 5 GB of free RAM).</>,
    code: `lightx-os --mode single --base-url http://localhost:11434/v1 --model qwen2.5:7b`,
  },
  {
    q: "lightx doctor finds a problem but offers no fix",
    a: <>Only patches that pass verification are offered. With no tests, a fix is checked only by re-running the check. Try a bigger model, more attempts, or skip dependency fixes.</>,
    code: `lightx doctor --fix --max-attempts 4
lightx doctor --fix --only code,migration`,
  },
  {
    q: "pip install lightagentx is missing SmartOS, the registry or doctor",
    a: <>You have an older release. Check the version and upgrade, or install straight from GitHub.</>,
    code: `python -c "import lightagentx; print(lightagentx.__version__)"
pip install -U "lightagentx @ git+${REPO}"`,
  },
];

function Issue({ q, a, code, open, onToggle }: (typeof ISSUES)[number] & { open: boolean; onToggle: () => void }) {
  return (
    <li className="rounded-xl border border-line bg-surface">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={open}
        className="flex min-h-12 w-full cursor-pointer items-center justify-between gap-4 px-5 py-4 text-left"
      >
        <span className="font-mono text-[13px] text-fg">{q}</span>
        <motion.span animate={{ rotate: open ? 180 : 0 }} transition={{ duration: 0.2 }} className="shrink-0 text-faint">
          <ChevronDown className="size-4" />
        </motion.span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease }}
            className="overflow-hidden"
          >
            <div className="space-y-3 px-5 pb-5 text-sm leading-relaxed text-muted">
              <p>{a}</p>
              {code && <CodeBlock code={code} lang={code.startsWith("agent") ? "python" : "bash"} title="fix" />}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </li>
  );
}

export function Setup() {
  return (
    <Section
      id="setup"
      eyebrow="Docs · Setup"
      title="Set everything up, step by step."
      lead="From an empty machine to a running agent, plus the optional pieces: providers, local models and sandbox isolation."
    >
      <ol className="space-y-6">
        {STEPS.map((s) => (
          <li key={s.title} className="grid gap-4 rounded-2xl border border-line bg-surface p-6 lg:grid-cols-[1fr_1.1fr] lg:items-start">
            <div>
              <h3 className="text-lg font-semibold">{s.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted">{s.body}</p>
            </div>
            {s.code && <CodeBlock code={s.code} lang={s.lang} title="terminal" />}
          </li>
        ))}
      </ol>
      <p className="mt-6 text-sm text-muted">
        Full reference:{" "}
        <a className="underline decoration-line-strong underline-offset-4 hover:text-fg" href={`${DOCS}/guide.md`} target="_blank" rel="noreferrer">
          the guide
        </a>
        .
      </p>
    </Section>
  );
}

export function Troubleshooting() {
  const [open, setOpen] = useState<number | null>(0);
  return (
    <Section
      id="troubleshooting"
      eyebrow="Docs · Troubleshooting"
      title="Something went wrong? Find the message here."
      lead="These are the actual error messages LightAgentX prints, what causes each one, and how to fix it."
    >
      <ul className="space-y-3">
        {ISSUES.map((it, i) => (
          <Issue key={it.q} {...it} open={open === i} onToggle={() => setOpen(open === i ? null : i)} />
        ))}
      </ul>
      <p className="mt-6 text-sm text-muted">
        Not listed?{" "}
        <a className="underline decoration-line-strong underline-offset-4 hover:text-fg" href={`${REPO}/issues`} target="_blank" rel="noreferrer">
          Open an issue
        </a>{" "}
        with the full error and <C>python -c "import lightagentx; print(lightagentx.__version__)"</C>.
      </p>
    </Section>
  );
}
