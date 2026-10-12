import { useEffect, useRef, useState } from "react";
import { motion, useInView, useReducedMotion } from "motion/react";
import { Pause, Play } from "lucide-react";
import { Panel, Section } from "./ui";
import { ease } from "../lib";

const CODE = `def run(self, user_input: str) -> str:
    self.memory.add_message("user", user_input)
    tool_schemas = self.registry.to_openai_schema()

    for iteration in range(1, self.max_iterations + 1):
        messages = self.memory.get_messages()
        response = self.llm.chat_with_tools(messages, tool_schemas)

        if response.has_tool_calls:
            self.memory.add_assistant_tool_calls(
                content=response.content, tool_calls=response.tool_calls)
            results = self.executor.execute_many(response.tool_calls)
            for result in results:
                self.memory.add_tool_message(
                    tool_call_id=result["tool_call_id"], content=result["content"])
            continue  # back to the LLM with the new context

        self.memory.add_message("assistant", response.content)
        return response.content

    return f"Max iterations ({self.max_iterations}) reached."`.split("\n");

const STEPS = [
  {
    n: "01",
    title: "Show the model everything",
    body: "The conversation so far and the JSON schema of every tool, generated from your type hints and docstrings.",
    lines: [1, 2, 4, 5, 6],
  },
  {
    n: "02",
    title: "The model answers or asks for tools",
    body: "A single LLMResponse type works across OpenAI, Anthropic and Gemini. The tool-call intent is saved first, which is the ordering rule the API enforces.",
    lines: [8, 9, 10],
  },
  {
    n: "03",
    title: "Run the tools, record what they returned",
    body: "The executor dispatches each call by name. Results go back into memory as tool messages, tied to their call IDs.",
    lines: [11, 12, 13, 14],
  },
  {
    n: "04",
    title: "Loop, or return the answer",
    body: "A flat for-loop with continue, capped at max_iterations. There's no recursion and no hidden callbacks.",
    lines: [15, 17, 18, 20],
  },
];

export function Loop() {
  const [active, setActive] = useState(0);
  const [paused, setPaused] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { margin: "-20%" });
  const reduce = useReducedMotion();

  useEffect(() => {
    if (paused || !inView || reduce) return;
    const t = setTimeout(() => setActive((a) => (a + 1) % STEPS.length), 3600);
    return () => clearTimeout(t);
  }, [active, paused, inView, reduce]);

  const hot = new Set(STEPS[active].lines);

  return (
    <Section
      id="loop"
      n="01"
      title="The loop everything else hangs off"
      lead={
        <p>
          An LLM on its own only produces text. To make it act, you show it the conversation and the tools, run whatever
          tools it asks for, add the results to the conversation and ask again. LangChain's <code className="ic">AgentExecutor</code>,
          CrewAI and AutoGPT all wrap this same loop. Here is the version LightAgentX runs, trimmed of logging.
        </p>
      }
    >
      <div ref={ref} className="grid gap-8 lg:grid-cols-[0.9fr_1.1fr]">
        <ol className="space-y-2">
          {STEPS.map((s, i) => {
            const on = i === active;
            return (
              <li key={s.n}>
                <button
                  type="button"
                  onClick={() => {
                    setActive(i);
                    setPaused(true);
                  }}
                  aria-current={on ? "step" : undefined}
                  className="relative w-full cursor-pointer rounded-lg p-5 text-left transition-colors hover:bg-surface-2/60"
                >
                  {on && (
                    <motion.div
                      layoutId="loop-active"
                      className="absolute inset-0 rounded-lg border border-line bg-surface"
                      transition={{ type: "spring", stiffness: 380, damping: 34 }}
                    />
                  )}
                  <div className="relative z-10 flex gap-4">
                    <span className={`font-mono text-sm ${on ? "text-accent" : "text-faint"}`}>{s.n}</span>
                    <div>
                      <h3 className={`text-lg font-medium ${on ? "text-fg" : "text-muted"}`}>{s.title}</h3>
                      <motion.p
                        initial={false}
                        animate={{ height: on ? "auto" : 0, opacity: on ? 1 : 0 }}
                        transition={{ duration: 0.35, ease }}
                        className="overflow-hidden text-[15px] leading-relaxed text-muted"
                      >
                        <span className="block pt-2">{s.body}</span>
                      </motion.p>
                    </div>
                  </div>
                  {on && !paused && inView && !reduce && (
                    <motion.div
                      key={active}
                      className="absolute bottom-0 left-5 h-px bg-accent"
                      initial={{ width: 0 }}
                      animate={{ width: "calc(100% - 2.5rem)" }}
                      transition={{ duration: 3.6, ease: "linear" }}
                    />
                  )}
                </button>
              </li>
            );
          })}
        </ol>

        <Panel
          title="lightagentx/loop/engine.py"
          action={
            !reduce && (
              <button
                type="button"
                onClick={() => setPaused((p) => !p)}
                aria-label={paused ? "Resume step walkthrough" : "Pause step walkthrough"}
                className="inline-flex size-11 cursor-pointer items-center justify-center rounded-md text-code-dim transition-colors duration-200 hover:bg-white/10 hover:text-code-fg md:size-9"
              >
                {paused ? <Play className="size-3.5" /> : <Pause className="size-3.5" />}
              </button>
            )
          }
        >
          <pre className="overflow-x-auto py-3 font-mono text-[12.5px] leading-[1.7]">
            {CODE.map((line, i) => (
              <motion.div
                key={i}
                animate={{
                  opacity: hot.has(i) ? 1 : 0.4,
                  backgroundColor: hot.has(i) ? "rgba(134,239,172,0.08)" : "rgba(0,0,0,0)",
                }}
                transition={{ duration: 0.3 }}
                className="flex min-h-[1.7em] pr-4"
              >
                <span className="w-10 shrink-0 pr-3 text-right text-code-dim select-none">{i + 1}</span>
                <motion.span
                  className="w-0.5 shrink-0"
                  animate={{ backgroundColor: hot.has(i) ? "#86efac" : "rgba(0,0,0,0)" }}
                />
                <span className="pl-3 whitespace-pre">{line}</span>
              </motion.div>
            ))}
          </pre>
        </Panel>
      </div>
    </Section>
  );
}
