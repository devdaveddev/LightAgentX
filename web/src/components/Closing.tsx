import { useState } from "react";
import { motion } from "motion/react";
import { Wordmark } from "./Nav";
import { CodeBlock, CopyButton, Section } from "./ui";
import { DOCS, PYPI, REPO, install } from "../lib";

const MAP: [string, string][] = [
  ["BaseLLM / OpenAILLM", "BaseChatModel / ChatOpenAI"],
  ["SmartRouter", "No equivalent (custom routing needed)"],
  ["BufferMemory", "ConversationBufferWindowMemory"],
  ["@tool / BaseTool", "@tool / StructuredTool"],
  ["AgentLoop", "AgentExecutor.invoke()"],
  ["ReActPlanner", "create_react_agent + ReActOutputParser"],
  ["SequentialPipeline", "SequentialChain / LCEL chain1 | chain2"],
  ["CrewAgent", "CrewAI Crew(process=hierarchical)"],
  ["HookRegistry", "Callbacks / BaseCallbackHandler"],
  ["Sandbox / SmartOS", "No equivalent (ShellTool runs unsandboxed)"],
  ["AgentRegistry", "LangGraph checkpointer + thread_id (no merge, no ACLs)"],
];

export function Equivalence() {
  return (
    <Section
      id="map"
      n="06"
      title="If you know LangChain"
      lead={<p>Most pieces have a direct counterpart. The rows without one are the parts I built because I needed them.</p>}
    >
      <div className="overflow-x-auto rounded-lg border border-line bg-surface">
        <table className="w-full min-w-[34rem] text-left text-[15px]">
          <thead className="border-b border-line text-sm text-faint">
            <tr>
              <th scope="col" className="px-5 py-3 font-medium">LightAgentX</th>
              <th scope="col" className="px-5 py-3 font-medium">LangChain</th>
            </tr>
          </thead>
          <tbody>
            {MAP.map(([a, b]) => (
              <tr key={a} className="border-t border-line first:border-t-0">
                <td className="px-5 py-3 font-mono text-[13px] text-accent">{a}</td>
                <td className={`px-5 py-3 ${b.startsWith("No equivalent") ? "text-faint italic" : "text-muted"}`}>{b}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Section>
  );
}

// From the "Limitations (honest)" sections of the docs and the README.
const LIMITS: [area: string, text: string][] = [
  ["Sandbox", "Real isolation (bubblewrap) is Linux-only. Elsewhere commands run as plain subprocesses and every one needs your approval."],
  ["Sandbox", "File tools, opening files and launching apps act on your real desktop. They're guarded by policy and confirmation, not isolation."],
  ["Registry", "Principals are names, not authenticated identities. It's meant for cooperating processes on one machine or a shared filesystem."],
  ["Registry", "History grows without limit; there's no garbage collection yet. Only SingleAgent is supported."],
  ["Registry", "Merging two diverged transcripts keeps them valid, but can't make two different conversations agree."],
  ["Doctor", "Verification is only as strong as your tests. Changes that don't fail at import time show up only through tests or warnings."],
  ["Doctor", "The \"where does it live now\" search is by name: it finds moved code, not renamed functions."],
];

export function Limits() {
  return (
    <Section
      id="limits"
      n="07"
      title="What it doesn't do"
      lead={<p>This is a learning project that grew real features. These are the edges I know about.</p>}
    >
      <dl className="grid gap-x-12 md:grid-cols-2">
        {LIMITS.map(([area, text], i) => (
          <div key={i} className="flex gap-4 border-t border-line py-4">
            <dt className="w-20 shrink-0 font-mono text-sm text-faint">{area}</dt>
            <dd className="leading-relaxed text-muted">{text}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-8 text-sm text-faint">
        Details:{" "}
        <a className="link" href={`${DOCS}/agent-registry.md`} target="_blank" rel="noreferrer">
          registry
        </a>
        ,{" "}
        <a className="link" href={`${DOCS}/doctor.md`} target="_blank" rel="noreferrer">
          doctor
        </a>
        .
      </p>
    </Section>
  );
}

const TABS = [
  {
    id: "tools",
    label: "Tool agent",
    code: `from lightagentx import OpenAILLM, SingleAgent, tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: Name of the city to get weather for.
    """
    return f"Weather in {city}: 22°C, Clear skies"

agent = SingleAgent(name="Assistant", llm=OpenAILLM(model="gpt-4o-mini"),
                    tools=[get_weather], verbose=True)
print(agent.run("What's the weather like in Tokyo?"))`,
  },
  {
    id: "hooks",
    label: "Hooks",
    code: `from lightagentx import HookRegistry, OpenAILLM, SingleAgent

hooks = HookRegistry()

@hooks.on("after_tool_call")
def track(event):
    print(event.data["tool_name"], "→", event.data["result"][:100])

agent = SingleAgent(name="Bot", llm=OpenAILLM(), hooks=hooks)
agent.run("Hello")`,
  },
  {
    id: "local",
    label: "Fully local",
    lang: "bash" as const,
    code: `${install("os")}

# SmartOS chat terminal on a local model
lightx-os --base-url http://localhost:11434/v1 --model llama3.1

# check and fix a project
lightx doctor --fix --base-url http://localhost:11434/v1 --model qwen2.5:3b`,
  },
];

export function Quickstart() {
  const [tab, setTab] = useState(TABS[0].id);
  const current = TABS.find((t) => t.id === tab)!;
  const steps: [string, string][] = [
    ["Install", install()],
    ["Set a key (or skip it and use a local model)", "export OPENAI_API_KEY=..."],
    ["Run an example from the repo", "python examples/02_tool_agent.py"],
  ];
  return (
    <Section id="quickstart" n="08" title="Trying it">
      <div className="grid gap-10 lg:grid-cols-[0.8fr_1.2fr]">
        <ol className="space-y-7">
          {steps.map(([t, c], i) => (
            <li key={t}>
              <div className="text-muted">
                <span className="font-mono text-sm text-faint">{i + 1}.</span> {t}
              </div>
              <div className="mt-2 flex items-center gap-2 rounded-lg border border-code-line bg-code py-1 pr-1 pl-3 text-code-fg">
                <code className="flex-1 overflow-x-auto font-mono text-xs whitespace-nowrap">{c}</code>
                <CopyButton text={c} />
              </div>
            </li>
          ))}
          <li className="text-sm leading-relaxed text-faint">
            Extras: <code className="ic">[anthropic]</code> <code className="ic">[gemini]</code> <code className="ic">[os]</code>{" "}
            <code className="ic">[all]</code>.
          </li>
        </ol>
        <div>
          <div role="tablist" aria-label="Code examples" className="mb-3 flex gap-1 border-b border-line">
            {TABS.map((t) => (
              <button
                key={t.id}
                role="tab"
                type="button"
                aria-selected={tab === t.id}
                aria-controls="qs-panel"
                onClick={() => setTab(t.id)}
                className={`relative min-h-11 cursor-pointer px-3 text-sm whitespace-nowrap transition-colors ${
                  tab === t.id ? "text-fg" : "text-muted hover:text-fg"
                }`}
              >
                {t.label}
                {tab === t.id && (
                  <motion.span layoutId="qs-tab" className="absolute inset-x-2 -bottom-px h-0.5 bg-accent" transition={{ type: "spring", stiffness: 500, damping: 40 }} />
                )}
              </button>
            ))}
          </div>
          <div id="qs-panel" role="tabpanel">
            <CodeBlock code={current.code} lang={current.lang ?? "python"} title={current.lang ? "terminal" : "agent.py"} />
          </div>
        </div>
      </div>
    </Section>
  );
}

export function Footer() {
  return (
    <footer className="border-t border-line">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 pt-12 pb-[calc(3rem+env(safe-area-inset-bottom))] sm:flex-row sm:items-end sm:justify-between sm:px-6">
        <div>
          <Wordmark />
          <p className="mt-2 max-w-md text-sm leading-relaxed text-faint">Built by Devansh to learn how agent frameworks work. MIT licensed.</p>
        </div>
        <p className="flex flex-wrap gap-x-5 gap-y-2 text-sm">
          <a className="link" href={REPO} target="_blank" rel="noreferrer">
            GitHub
          </a>
          <a className="link" href={`${DOCS}/guide.md`} target="_blank" rel="noreferrer">
            Guide
          </a>
          <a className="link" href={PYPI} target="_blank" rel="noreferrer">
            PyPI
          </a>
          <a className="link" href={`${REPO}/issues`} target="_blank" rel="noreferrer">
            Issues
          </a>
        </p>
      </div>
    </footer>
  );
}
