import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { ArrowRight } from "lucide-react";
import { Logo } from "./Nav";
import { CodeBlock, CopyButton, GitHubIcon, Reveal, Section } from "./ui";
import { DOCS, REPO } from "../lib";

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
      eyebrow="Coming from LangChain?"
      title="The same ideas, without the layers on top."
      lead="If you know LangChain, you already know the shape of LightAgentX. The difference is that you can read all of it."
    >
      <Reveal className="overflow-hidden rounded-2xl border border-line">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[34rem] text-left text-sm">
            <thead className="bg-surface text-xs tracking-wider text-faint uppercase">
              <tr>
                <th scope="col" className="px-5 py-3 font-medium">LightAgentX</th>
                <th scope="col" className="px-5 py-3 font-medium">LangChain</th>
              </tr>
            </thead>
            <tbody>
              {MAP.map(([a, b]) => (
                <tr key={a} className="border-t border-line transition-colors hover:bg-white/[0.02]">
                  <td className="px-5 py-3 font-mono text-[13px] text-accent">{a}</td>
                  <td className={`px-5 py-3 ${b.startsWith("No equivalent") ? "text-faint italic" : "text-muted"}`}>{b}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Reveal>
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
    id: "crew",
    label: "Crew",
    code: `from lightagentx import CrewAgent, OpenAILLM, SingleAgent

llm = OpenAILLM()
researcher = SingleAgent(name="Researcher", llm=llm, tools=[search])
writer = SingleAgent(name="Writer", llm=llm)

crew = CrewAgent(name="Newsroom", agents=[researcher, writer], manager_llm=llm)
crew.run("Write a short brief on small local LLMs")`,
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
    code: `pip install "lightagentx[os]"

# SmartOS chat terminal on a local model
lightx-os --base-url http://localhost:11434/v1 --model llama3.1

# or check and fix a project
lightx doctor --fix --base-url http://localhost:11434/v1 --model qwen2.5:3b`,
  },
];

export function Quickstart() {
  const [tab, setTab] = useState(TABS[0].id);
  const current = TABS.find((t) => t.id === tab)!;
  return (
    <Section id="quickstart" eyebrow="Quickstart" title="From pip install to a working agent in about a minute.">
      <div className="grid gap-8 lg:grid-cols-[0.75fr_1.25fr]">
        <Reveal className="space-y-6">
          {[
            ["Install", 'pip install lightagentx   # extras: [anthropic] [gemini] [os] [all]'],
            ["Set a key", "export OPENAI_API_KEY=...   # or ANTHROPIC / GOOGLE, or none for local"],
            ["Run an example", "python examples/02_tool_agent.py"],
          ].map(([t, c], i) => (
            <div key={t} className="flex gap-4">
              <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-accent font-mono text-sm font-semibold text-accent-ink">
                {i + 1}
              </span>
              <div className="min-w-0 flex-1">
                <div className="font-medium">{t}</div>
                <div className="mt-2 flex items-center gap-2 rounded-lg border border-line bg-surface py-1 pr-1 pl-3">
                  <code className="flex-1 overflow-x-auto font-mono text-xs whitespace-nowrap text-muted">{c}</code>
                  <CopyButton text={c.split("   #")[0]} />
                </div>
              </div>
            </div>
          ))}
        </Reveal>
        <Reveal delay={0.1}>
          <div role="tablist" aria-label="Code examples" className="mb-3 flex gap-1 overflow-x-auto rounded-lg border border-line bg-surface p-1">
            {TABS.map((t) => (
              <button
                key={t.id}
                role="tab"
                type="button"
                aria-selected={tab === t.id}
                aria-controls="qs-panel"
                onClick={() => setTab(t.id)}
                className={`relative min-h-11 flex-1 cursor-pointer rounded-md px-3 text-sm whitespace-nowrap transition-colors ${
                  tab === t.id ? "text-fg" : "text-muted hover:text-fg"
                }`}
              >
                {tab === t.id && (
                  <motion.span
                    layoutId="qs-tab"
                    className="absolute inset-0 rounded-md bg-surface-2 ring-1 ring-line-strong"
                    transition={{ type: "spring", stiffness: 420, damping: 34 }}
                  />
                )}
                <span className="relative">{t.label}</span>
              </button>
            ))}
          </div>
          <div id="qs-panel" role="tabpanel">
            <AnimatePresence mode="wait">
              <motion.div
                key={tab}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6, transition: { duration: 0.12 } }}
                transition={{ duration: 0.25 }}
              >
                <CodeBlock code={current.code} lang={current.lang ?? "python"} title={current.lang ? "terminal" : "agent.py"} />
              </motion.div>
            </AnimatePresence>
          </div>
        </Reveal>
      </div>
    </Section>
  );
}

export function Footer() {
  return (
    <footer className="relative mt-8 border-t border-line">
      <div className="mx-auto max-w-6xl px-4 py-20 text-center sm:px-6">
        <Reveal>
          <h2 className="mx-auto max-w-2xl text-[1.7rem] leading-tight font-semibold text-balance sm:text-[2.15rem]">
            Read the code you depend on.
          </h2>
          <p className="mx-auto mt-4 max-w-lg text-muted">MIT licensed and written in pure Python. Clone it, read it, change it.</p>
          <div className="mt-8 flex flex-col justify-center gap-3 sm:flex-row">
            <a
              href={REPO}
              target="_blank"
              rel="noreferrer"
              className="inline-flex h-12 items-center justify-center gap-2 rounded-lg bg-accent px-6 font-medium text-accent-ink transition-transform hover:-translate-y-0.5"
            >
              <GitHubIcon /> Star on GitHub
            </a>
            <a
              href={`${DOCS}/guide.md`}
              className="inline-flex h-12 items-center justify-center gap-2 rounded-lg border border-line-strong px-6 font-medium transition-colors hover:bg-white/5"
            >
              Read the guide <ArrowRight className="size-4" />
            </a>
          </div>
        </Reveal>
      </div>
      <div className="border-t border-line">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 px-4 pt-8 pb-[calc(2rem+env(safe-area-inset-bottom))] text-sm text-faint sm:flex-row sm:px-6">
          <Logo />
          <p>Built from scratch to understand how agents work under the hood.</p>
        </div>
      </div>
    </footer>
  );
}
