import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion, useInView, useReducedMotion } from "motion/react";
import { ArrowRight, Pause, Play } from "lucide-react";
import { CopyButton, GitHubIcon } from "./ui";
import { REPO, ease, fadeUp, stagger } from "../lib";

type Line = { kind: "user" | "think" | "call" | "result" | "answer"; text: string };

const TRACE: Line[] = [
  { kind: "user", text: "What's the weather in Tokyo, and what's 25 * 37 + 12?" },
  { kind: "think", text: "Two independent facts needed → call both tools." },
  { kind: "call", text: "get_weather(city='Tokyo')" },
  { kind: "call", text: "calculate(expression='25 * 37 + 12')" },
  { kind: "result", text: "Weather in Tokyo: 22°C, Clear skies" },
  { kind: "result", text: "937" },
  { kind: "answer", text: "It's 22°C and clear in Tokyo, and 25 × 37 + 12 = 937." },
];

const STYLE: Record<Line["kind"], { tag: string; cls: string }> = {
  user: { tag: "you", cls: "text-fg" },
  think: { tag: "think", cls: "text-violet" },
  call: { tag: "tool", cls: "text-sky" },
  result: { tag: "obs", cls: "text-muted" },
  answer: { tag: "agent", cls: "text-accent" },
};

function Terminal() {
  const reduce = useReducedMotion();
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref);
  const [playing, setPlaying] = useState(!reduce);
  const [shown, setShown] = useState(reduce ? TRACE.length : 0);

  // Autoplay only while visible and not paused; reduced motion gets the finished trace.
  useEffect(() => {
    if (reduce || !playing || !inView) return;
    const done = shown >= TRACE.length;
    const t = setTimeout(() => setShown(done ? 0 : shown + 1), done ? 3200 : shown === 0 ? 500 : 850);
    return () => clearTimeout(t);
  }, [shown, reduce, playing, inView]);

  return (
    <div ref={ref} className="relative">
      <div className="absolute -inset-px rounded-2xl bg-gradient-to-b from-accent/30 via-sky/10 to-transparent opacity-60 blur-xl" aria-hidden />
      <div className="relative overflow-hidden rounded-2xl border border-line-strong bg-surface/95 shadow-2xl shadow-black/50">
        <div className="flex items-center gap-2 border-b border-line px-4 py-3">
          <span className="size-3 rounded-full bg-rose/70" />
          <span className="size-3 rounded-full bg-amber/70" />
          <span className="size-3 rounded-full bg-accent/70" />
          <span className="ml-3 flex-1 truncate font-mono text-xs text-faint">python agent.py — AgentLoop</span>
          {!reduce && (
            <button
              type="button"
              onClick={() => setPlaying((p) => !p)}
              aria-label={playing ? "Pause demo" : "Play demo"}
              className="-my-2.5 -mr-2 inline-flex size-11 cursor-pointer md:-my-1.5 md:size-9 items-center justify-center rounded-md text-faint transition-colors duration-200 hover:bg-white/5 hover:text-fg"
            >
              {playing ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
            </button>
          )}
        </div>
        <div className="h-[22rem] space-y-2.5 p-4 font-mono text-[13px] leading-relaxed sm:p-5" role="log" aria-label="Example agent trace">
          <AnimatePresence initial={false}>
            {TRACE.slice(0, shown).map((l, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, transition: { duration: 0.2 } }}
                transition={{ duration: 0.35, ease }}
                className="flex gap-3"
              >
                <span className="w-12 shrink-0 text-right text-faint">{STYLE[l.kind].tag}</span>
                <span className={STYLE[l.kind].cls}>
                  {l.kind === "call" && <span className="text-faint">→ </span>}
                  {l.text}
                </span>
              </motion.div>
            ))}
          </AnimatePresence>
          {shown < TRACE.length && playing && (
            <motion.span
              className="ml-15 inline-block h-4 w-2 bg-accent/80 align-middle"
              animate={{ opacity: [1, 0, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
              aria-hidden
            />
          )}
        </div>
      </div>
    </div>
  );
}

const INSTALL = "pip install lightagentx";

export function Hero() {
  return (
    <section id="top" className="relative overflow-hidden pt-[calc(8rem+env(safe-area-inset-top))] pb-20 md:pt-[calc(10rem+env(safe-area-inset-top))] md:pb-28">
      <div className="bg-grid pointer-events-none absolute inset-0" aria-hidden />
      <motion.div
        className="pointer-events-none absolute top-[-10rem] left-1/2 h-[30rem] w-[60rem] -translate-x-1/2 rounded-full bg-accent/10 blur-3xl"
        animate={{ opacity: [0.5, 0.8, 0.5] }}
        transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
        aria-hidden
      />
      <div className="relative mx-auto grid max-w-6xl items-center gap-14 px-4 sm:px-6 lg:grid-cols-[1.05fr_1fr]">
        <motion.div variants={stagger(0.1)} initial="hidden" animate="show">
          <motion.a
            variants={fadeUp}
            href="#doctor"
            className="inline-flex items-center gap-2 rounded-full border border-line-strong bg-surface/60 py-1 pr-3 pl-1 text-sm text-muted transition-colors hover:text-fg"
          >
            <span className="rounded-full bg-accent px-2 py-0.5 text-xs font-semibold text-accent-ink">New</span>
            lightx doctor: verified fixes for broken upgrades
            <ArrowRight className="size-3.5" />
          </motion.a>
          <motion.h1
            variants={fadeUp}
            className="mt-6 text-[2.6rem] leading-[1.08] font-semibold text-balance sm:text-[3.4rem]"
          >
            Agents, without <span className="glow text-accent">the magic.</span>
          </motion.h1>
          <motion.p variants={fadeUp} className="mt-6 max-w-xl text-lg leading-relaxed text-pretty text-muted">
            LightAgentX is an agent framework written from scratch in pure Python. You get tool calling,
            memory, ReAct planning and multi-agent crews, plus a real sandbox, versioned agents and
            verified fixes. Every line is meant to be read.
          </motion.p>
          <motion.div variants={fadeUp} className="mt-9 flex flex-col gap-3 sm:flex-row sm:items-center">
            <a
              href="#quickstart"
              className="inline-flex h-12 items-center justify-center gap-2 rounded-lg bg-accent px-6 font-medium text-accent-ink transition-transform hover:-translate-y-0.5 active:translate-y-0"
            >
              Get started <ArrowRight className="size-4" />
            </a>
            <div className="flex h-12 items-center gap-3 rounded-lg border border-line-strong bg-surface pr-1.5 pl-4 font-mono text-sm">
              <span className="text-faint select-none">$</span>
              <span className="flex-1">{INSTALL}</span>
              <CopyButton text={INSTALL} />
            </div>
          </motion.div>
          <motion.div variants={fadeUp} className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-faint">
            <span>Python 3.10+</span>
            <span aria-hidden>·</span>
            <span>MIT licensed</span>
            <span aria-hidden>·</span>
            <a href={REPO} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 hover:text-fg">
              <GitHubIcon className="size-3.5" /> devdaveddev/LightAgentX
            </a>
          </motion.div>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 32, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ duration: 0.8, ease, delay: 0.3 }}
        >
          <Terminal />
        </motion.div>
      </div>
    </section>
  );
}
