import { CodeBlock, Section } from "./ui";
import { REPO, SRC } from "../lib";

// The real top level of lightagentx/, in the order the README walks through it.
const TREE: [name: string, what: string][] = [
  ["llm/", "One interface over OpenAI, Anthropic, Gemini and local servers, plus SmartRouter"],
  ["memory/", "Conversation history: a sliding window, or older turns summarised by the LLM"],
  ["tools/", "The @tool decorator, a name-based registry and the executor"],
  ["planning/", "ReAct: Thought → Action → Observation"],
  ["loop/", "The agent loop from section 01"],
  ["agents/", "SingleAgent, SequentialPipeline, CrewAgent"],
  ["hooks.py", "Callbacks for eight lifecycle events"],
  ["snapshot.py", "Save an agent with its memory and config; restore it elsewhere"],
  ["sandbox/", "Policy, isolation, confirmation and audit (section 03)"],
  ["smartos/", "A crew of agents that manage your OS, and the lightx-os terminal"],
  ["state/", "The agent registry: versioned, permissioned agents (section 04)"],
  ["doctor/", "lightx doctor (section 05)"],
  ["utils/", "The colored logger that printed the trace at the top"],
];

function Tree() {
  return (
    <div className="overflow-x-auto rounded-lg border border-line bg-surface">
      <ul className="py-2 font-mono text-[13px]">
        <li className="px-4 pb-1 text-faint">
          <a className="link" href={SRC} target="_blank" rel="noreferrer">
            lightagentx/
          </a>
        </li>
        {TREE.map(([name, what], i) => (
          <li key={name}>
            <a
              href={name.endsWith(".py") ? `${REPO}/blob/main/lightagentx/${name}` : `${SRC}/${name.slice(0, -1)}`}
              target="_blank"
              rel="noreferrer"
              className="group flex min-h-10 items-baseline gap-3 px-4 py-1.5 transition-colors hover:bg-surface-2"
            >
              <span className="shrink-0 text-faint select-none">{i === TREE.length - 1 ? "└──" : "├──"}</span>
              <span className="w-24 shrink-0 text-accent group-hover:underline sm:w-28">{name}</span>
              <span className="font-sans text-[14px] text-muted">{what}</span>
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function Modules() {
  return (
    <Section
      id="modules"
      n="02"
      title="What's in the box"
      lead={
        <p>
          Each folder has a <code className="ic">base.py</code> that defines its contract and one or more implementations, so any
          part can be swapped out. The tree below is the real package; every line links to its source.
        </p>
      }
    >
      <div className="grid gap-8 lg:grid-cols-[1.15fr_0.85fr]">
        <Tree />
        <div className="space-y-6">
          <CodeBlock
            title="tools: type hints become the JSON schema"
            code={`@tool(risk="low", reads=["path"])
def read_report(path: str) -> str:
    """Read a report from disk.

    Args:
        path: Where the report is.
    """`}
          />
          <CodeBlock
            title="llm: mix providers in one router"
            code={`router = SmartRouter(
    models=[OpenAILLM(base_url="http://localhost:11434/v1", model="llama3.1",
                      api_key="local"),   # any OpenAI-compatible server
            AnthropicLLM(), GeminiLLM()],
    strategy="complexity",   # or round_robin, fallback, cost_limit
)
agent = SingleAgent(name="Bot", llm=router)`}
          />
          <p className="text-sm leading-relaxed text-faint">
            Full walkthrough of every module, with the code: the{" "}
            <a className="link" href={`${REPO}#readme`} target="_blank" rel="noreferrer">
              README
            </a>
            .
          </p>
        </div>
      </div>
    </Section>
  );
}
