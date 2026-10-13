import type { ReactNode } from "react";
import { motion, useMotionTemplate, useMotionValue } from "motion/react";
import { Brain, Camera, GitFork, Layers, Route, Users, Webhook, Wrench } from "lucide-react";
import type { Variants } from "motion/react";
import { Section } from "./ui";
import { stagger } from "../lib";

// Grid stagger with a slight overshoot (UI UX Pro Max "Stagger List"); fine for cards, not for tables.
const cardIn: Variants = {
  hidden: { opacity: 0, y: 16, scale: 0.94 },
  show: { opacity: 1, y: 0, scale: 1, transition: { type: "spring", stiffness: 260, damping: 20 } },
};

function Card({
  icon,
  title,
  path,
  children,
  className = "",
}: {
  icon: ReactNode;
  title: string;
  path: string;
  children: ReactNode;
  className?: string;
}) {
  const x = useMotionValue(-400);
  const y = useMotionValue(-400);
  const glow = useMotionTemplate`radial-gradient(420px circle at ${x}px ${y}px, rgba(34,197,94,0.09), transparent 60%)`;
  return (
    <motion.article
      variants={cardIn}
      onPointerMove={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        x.set(e.clientX - r.left);
        y.set(e.clientY - r.top);
      }}
      onPointerLeave={() => {
        x.set(-400);
        y.set(-400);
      }}
      whileHover={{ y: -3 }}
      transition={{ type: "spring", stiffness: 300, damping: 26 }}
      className={`group relative overflow-hidden rounded-2xl border border-line bg-surface p-6 transition-colors hover:border-line-strong ${className}`}
    >
      <motion.div className="pointer-events-none absolute inset-0" style={{ background: glow }} aria-hidden />
      <div className="relative">
        <div className="flex items-center justify-between">
          <span className="inline-flex size-10 items-center justify-center rounded-lg border border-line bg-surface-2 text-accent">
            {icon}
          </span>
          <code className="font-mono text-xs text-faint">{path}</code>
        </div>
        <h3 className="mt-5 text-lg font-semibold tracking-tight">{title}</h3>
        <div className="mt-2 text-sm leading-relaxed text-muted">{children}</div>
      </div>
    </motion.article>
  );
}

function Mini({ children }: { children: ReactNode }) {
  return (
    <pre className="mt-4 overflow-x-auto rounded-lg border border-line bg-bg/60 p-3 font-mono text-[12px] leading-relaxed text-fg/90">
      {children}
    </pre>
  );
}

export function Modules() {
  return (
    <Section
      id="modules"
      eyebrow="Modules"
      title="Small parts, each with a clear contract."
      lead="Every folder has a base.py that defines the contract and one or more implementations. Use only the parts you need, and swap any of them out."
    >
      <motion.div
        variants={stagger(0.06)}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, margin: "-80px" }}
        className="grid gap-4 md:grid-cols-6"
      >
        <Card icon={<Wrench className="size-5" />} title="@tool: write a function, get a tool" path="tools/" className="md:col-span-3">
          The JSON schema is generated from your type hints and docstring. Tools can declare a risk level and the paths they read, so the sandbox can check them.
          <Mini>
            <span className="tk-dec">@tool</span>(risk=<span className="tk-str">"low"</span>, reads=[<span className="tk-str">"path"</span>]){"\n"}
            <span className="tk-kw">def</span> <span className="tk-fn">read_report</span>(path: str) -&gt; str:{"\n"}
            {"    "}<span className="tk-str">"""Read a report from disk."""</span>
          </Mini>
        </Card>
        <Card icon={<Route className="size-5" />} title="SmartRouter: the right model for each request" path="llm/router.py" className="md:col-span-3">
          Pool models from any provider, such as a local Ollama model, Claude and Gemini. Simple questions go to the cheap one and hard ones go to the strong one. It is itself a BaseLLM, so it works anywhere an LLM does.
          <Mini>
            router = <span className="tk-cls">SmartRouter</span>(models=[ollama, claude, gemini],{"\n"}
            {"                     "}strategy=<span className="tk-str">"complexity"</span>){"\n"}
            <span className="tk-com"># complexity · round_robin · fallback · cost_limit</span>
          </Mini>
        </Card>
        <Card icon={<Users className="size-5" />} title="Single, Sequential, Crew" path="agents/" className="md:col-span-2">
          One agent with tools, a pipeline that passes output from one agent to the next, or a manager LLM that hands work to specialists.
        </Card>
        <Card icon={<Brain className="size-5" />} title="Memory" path="memory/" className="md:col-span-2">
          BufferMemory keeps a sliding window. SummaryMemory has the LLM compress older turns so long chats stay within budget.
        </Card>
        <Card icon={<Layers className="size-5" />} title="ReAct planning" path="planning/" className="md:col-span-2">
          A Thought → Action → Observation loop with a parser that is easy to read and step through in a debugger.
        </Card>
        <Card icon={<Webhook className="size-5" />} title="Lifecycle hooks" path="hooks.py" className="md:col-span-2">
          Eight events, from before_llm_call to on_agent_end. Use them for logging, guardrails, cost tracking or rate limits, with no monkey-patching.
        </Card>
        <Card icon={<Camera className="size-5" />} title="Snapshots" path="snapshot.py" className="md:col-span-2">
          Export an agent with its memory, config and tool list, then restore it in another project. You can also export only the persona, which is safe to share.
        </Card>
        <Card icon={<GitFork className="size-5" />} title="Any provider" path="llm/" className="md:col-span-2">
          OpenAI, Anthropic, Gemini, or any OpenAI-compatible local server such as Ollama. The rest of your code stays the same.
        </Card>
      </motion.div>
    </Section>
  );
}
