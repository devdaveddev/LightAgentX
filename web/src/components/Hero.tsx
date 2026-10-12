import { CopyButton, Panel } from "./ui";
import { DOCS, PYPI, REPO, install } from "../lib";

type Line = { tag?: keyof typeof TAG; text: string; detail?: string[] };

const TAG = {
  SYSTEM: "ansi-cyan",
  AGENT: "ansi-cyan",
  THOUGHT: "ansi-yellow",
  ACTION: "ansi-magenta",
  OBSERVATION: "ansi-blue",
  RESULT: "ansi-green",
} as const;

// Captured from SingleAgent(verbose=True) with a scripted model in place of the LLM.
const TRACE: Line[] = [
  { tag: "SYSTEM", text: "AgentLoop initialized", detail: ["Model: Scripted(model='scripted')", "Tools: ['get_weather', 'calculate']", "Max iterations: 10"] },
  { tag: "AGENT", text: "[Assistant] Starting task: What's the weather in Tokyo, and what's 25 * 37 + 12?" },
  { text: "─".repeat(60) },
  { tag: "SYSTEM", text: "User: What's the weather in Tokyo, and what's 25 * 37 + 12?" },
  { tag: "SYSTEM", text: "Iteration 1/10" },
  { tag: "THOUGHT", text: "LLM requested tool calls" },
  { tag: "ACTION", text: "Calling `get_weather`", detail: ["{'city': 'Tokyo'}"] },
  { tag: "OBSERVATION", text: "Tool returned:", detail: ["22°C, Clear skies"] },
  { tag: "ACTION", text: "Calling `calculate`", detail: ["{'expression': '25 * 37 + 12'}"] },
  { tag: "OBSERVATION", text: "Tool returned:", detail: ["937"] },
  { tag: "SYSTEM", text: "Iteration 2/10" },
  { tag: "RESULT", text: "It's 22°C with clear skies in Tokyo, and 25 × 37 + 12 = 937." },
  { tag: "AGENT", text: "[Assistant] Task complete" },
];

function Trace() {
  return (
    <figure>
      <Panel title="python agent.py">
        <pre className="overflow-x-auto p-4 font-mono text-[12.5px] leading-[1.65]" aria-label="Agent trace">
          {TRACE.map((l, i) => (
            <div key={i} className={i ? "mt-2.5" : ""}>
              {l.tag ? (
                <>
                  <span className={`font-bold ${TAG[l.tag]}`}>[{l.tag}]</span> {l.text}
                </>
              ) : (
                <span className="text-code-dim">{l.text}</span>
              )}
              {l.detail?.map((d) => (
                <div key={d} className="pl-[3ch] text-code-dim">
                  {d}
                </div>
              ))}
            </div>
          ))}
        </pre>
      </Panel>
      <figcaption className="mt-3 text-sm leading-relaxed text-faint">
        Actual output of <code className="ic">SingleAgent(verbose=True)</code>. A scripted model stands in for the LLM so the run
        is reproducible; everything else is the real loop, tools and logger.
      </figcaption>
    </figure>
  );
}

export function Hero() {
  const cmd = install();
  return (
    <section id="top" className="mx-auto max-w-6xl px-4 pt-[calc(7rem+env(safe-area-inset-top))] pb-20 sm:px-6 md:pt-[calc(9rem+env(safe-area-inset-top))] md:pb-28">
      <div className="grid items-start gap-14 lg:grid-cols-[1fr_1.05fr]">
        <div>
          <p className="font-mono text-sm text-faint">v0.2.0 · Python 3.10+ · MIT</p>
          <h1 className="mt-5 text-[2.7rem] leading-[1.05] font-medium tracking-tight sm:text-6xl">LightAgentX</h1>
          <p className="mt-5 font-serif text-2xl leading-snug text-pretty text-muted sm:text-[1.7rem]">
            An agent framework written from scratch in plain Python, to understand how LangChain, CrewAI and AutoGPT work underneath.
          </p>
          <div className="mt-8 max-w-[60ch] space-y-4 leading-relaxed text-pretty text-muted">
            <p>
              I wanted to see what actually happens between a prompt and a tool call, so I wrote each piece myself: the loop,
              tool schemas, memory, ReAct planning and multi-agent crews. Then I kept going. Agents get a sandbox before they touch
              the OS, a registry so they outlive the process that started them, and a doctor for when a library upgrade breaks
              everything.
            </p>
            <p>
              The only required dependency is <code className="ic">openai</code>. Anthropic, Gemini and local models are optional
              extras.
            </p>
          </div>
          <div className="mt-9 flex items-center gap-2 rounded-lg border border-code-line bg-code py-1 pr-1 pl-4 font-mono text-[13px] text-code-fg">
            <span className="text-code-dim select-none">$</span>
            <span className="flex-1 overflow-x-auto whitespace-nowrap">{cmd}</span>
            <CopyButton text={cmd} />
          </div>
          <p className="mt-5 flex flex-wrap gap-x-5 gap-y-2 text-sm">
            <a className="link" href={REPO} target="_blank" rel="noreferrer">
              Source on GitHub
            </a>
            <a className="link" href={`${DOCS}/guide.md`} target="_blank" rel="noreferrer">
              The guide
            </a>
            <a className="link" href={PYPI} target="_blank" rel="noreferrer">
              PyPI
            </a>
          </p>
        </div>
        <Trace />
      </div>
    </section>
  );
}
