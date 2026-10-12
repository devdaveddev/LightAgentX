import { motion } from "motion/react";
import { CodeBlock, Reveal, Section } from "./ui";
import { DOCS, ease } from "../lib";

const PIPELINE = [
  { k: "PROPOSE", d: "The LLM gets the error, your code around the line, the installed version, and where the missing name lives now." },
  { k: "APPLY", d: "The patch is applied to a fresh copy of the project. Your own files aren't touched." },
  { k: "VERIFY", d: "In the sandbox, the problem must be gone and every import and test that passed before must still pass." },
  { k: "SHOW", d: "You see the diff and the verification result. If verification fails, the reason goes back to the LLM for another try." },
  { k: "APPLY?", d: "Nothing changes unless you answer y. The original is backed up first." },
];

const CHECKS = [
  ["Dependencies", "Missing or wrong versions, and pip conflicts"],
  ["Imports", "Names broken by an upgrade, and where they live now"],
  ["Deprecations", "Warnings raised in your code, with file and line"],
  ["Saved agents", "Old snapshot formats and corrupted registry history"],
];

export function Doctor() {
  return (
    <Section
      id="doctor"
      eyebrow="lightx doctor"
      title="When an upgrade breaks your agents, get a fix that's been tested first."
      lead="A function moves, a package splits in two, or an API gets deprecated. Doctor finds the problem. With --fix it checks a patch against your tests before offering it to you."
    >
      <div className="grid gap-6 lg:grid-cols-[1fr_1.1fr]">
        <Reveal className="flex flex-col gap-6">
          <CodeBlock
            lang="bash"
            title="terminal"
            code={`lightx doctor          # report only; changes nothing
lightx doctor --fix    # propose → verify in sandbox → you approve

# with a local model, nothing leaves your machine
lightx doctor --fix --base-url http://localhost:11434/v1 --model qwen2.5:3b`}
          />
          <div className="grid grid-cols-2 gap-3">
            {CHECKS.map(([t, d]) => (
              <div key={t} className="rounded-xl border border-line bg-surface p-4">
                <div className="text-sm font-medium">{t}</div>
                <div className="mt-1 text-xs leading-relaxed text-muted">{d}</div>
              </div>
            ))}
          </div>
        </Reveal>

        <ol className="relative">
          <motion.div
            className="absolute top-5 bottom-5 left-[19px] w-px origin-top bg-gradient-to-b from-accent to-sky"
            initial={{ scaleY: 0 }}
            whileInView={{ scaleY: 1 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 1.4, ease }}
            aria-hidden
          />
          {PIPELINE.map((p, i) => (
            <motion.li
              key={p.k}
              initial={{ opacity: 0, x: 16 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, margin: "-100px" }}
              transition={{ duration: 0.5, delay: 0.15 + i * 0.22, ease }}
              className="relative flex gap-5 pb-7 last:pb-0"
            >
              <span className="relative z-10 flex size-10 shrink-0 items-center justify-center rounded-full border border-line-strong bg-surface font-mono text-xs text-accent">
                {i + 1}
              </span>
              <div className="pt-1.5">
                <div className="font-mono text-sm tracking-wide">{p.k}</div>
                <p className="mt-1 text-sm leading-relaxed text-muted">{p.d}</p>
              </div>
            </motion.li>
          ))}
          <motion.li
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 1.5 }}
            className="mt-6 ml-15 list-none rounded-lg border border-line bg-surface-2 px-4 py-3 text-sm text-muted"
          >
            Then it checks again, because fixing one problem can uncover the next one. Patches may not edit your tests, so a fix can't pass by changing them.
          </motion.li>
        </ol>
      </div>
      <Reveal className="mt-6">
        <a href={`${DOCS}/doctor.md`} className="text-sm text-muted underline decoration-line-strong underline-offset-4 hover:text-fg">
          How detection and verification work →
        </a>
      </Reveal>
    </Section>
  );
}
