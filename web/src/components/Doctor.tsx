import { CodeBlock, Panel, Section } from "./ui";
import { DOCS } from "../lib";

const PIPELINE: [step: string, what: string][] = [
  ["Propose", "The LLM gets the error, your code around the line, the installed version and where the missing name lives now."],
  ["Apply", "The patch goes onto a fresh copy of the project, never your files."],
  ["Verify", "In the sandbox: the problem must be gone, and every import and test that passed before must still pass."],
  ["Show", "You see the diff and the verification result. A rejected patch goes back to the LLM with the reason."],
  ["Apply?", "Only if you answer y. The original is backed up first."],
];

export function Doctor() {
  return (
    <Section
      id="doctor"
      n="05"
      title="When an upgrade breaks things"
      lead={
        <p>
          Library upgrades break agent projects all the time: a function moves, a package splits in two, an API is deprecated.{" "}
          <code className="ic">lightx doctor</code> finds these. With <code className="ic">--fix</code> it asks an LLM for a patch,
          proves the patch works in the sandbox, and changes your code only after you say yes.
        </p>
      }
    >
      <div className="grid gap-10 lg:grid-cols-2">
        <div className="space-y-6">
          <Panel title="lightx doctor  (the example from the test suite)">
            <pre className="overflow-x-auto p-4 font-mono text-[12.5px] leading-[1.7]">
              <span className="font-bold">Imports</span>
              {"\n  "}
              <span className="ansi-red">✗</span> 'load_data' no longer exists in fakelib  app.py:2  <span className="ansi-yellow">[fixable]</span>
              {"\n      "}
              <span className="text-code-dim">fakelib is installed; 'load_data' is now defined in: fakelib.io</span>
              {"\n  "}
              <span className="ansi-red">✗</span> Can't import fakelib.legacy  app.py:3  <span className="ansi-yellow">[fixable]</span>
              {"\n      "}
              <span className="text-code-dim">ModuleNotFoundError: No module named 'fakelib.legacy'</span>
              {"\n      "}
              <span className="text-code-dim">Possibly moved to: fakelib_community.legacy</span>
            </pre>
          </Panel>
          <p className="leading-relaxed text-muted">
            With a local <code className="ic">qwen2.5:3b</code>, <code className="ic">--fix</code> proposed and verified{" "}
            <code className="ic">from fakelib.io import load_data</code> and{" "}
            <code className="ic">from fakelib_community.legacy import helper</code>. Once the imports worked the tests could run,
            which exposed a deprecation warning on line 8; that was fixed next with <code className="ic">compute_v2()</code>.
          </p>
          <CodeBlock
            lang="bash"
            title="terminal"
            code={`lightx doctor           # report only; changes nothing
lightx doctor --fix     # propose, verify in the sandbox, ask you

# a 3B model on a laptop CPU is enough
lightx doctor --fix --base-url http://127.0.0.1:11434/v1 --model qwen2.5:3b`}
          />
        </div>

        <div>
          <h3 className="text-xl font-medium">What --fix does with each problem</h3>
          <ol className="mt-5 border-l border-line">
            {PIPELINE.map(([step, what], i) => (
              <li key={step} className="relative pb-6 pl-6 last:pb-0">
                <span className="absolute top-1.5 -left-[5px] size-[9px] rounded-full border-2 border-accent bg-bg" aria-hidden />
                <div className="font-mono text-sm">
                  <span className="text-faint">{i + 1}.</span> {step}
                </div>
                <p className="mt-1 leading-relaxed text-muted">{what}</p>
              </li>
            ))}
          </ol>
          <p className="mt-8 leading-relaxed text-muted">
            Then it checks again, because fixing one problem can reveal the next. Patches may not touch your tests, so a fix can't
            pass by editing them. Small models are enough because the doctor has already found the problem, the line and where the
            name lives now; the model only writes a one-line edit, and wrong answers are cheap.
          </p>
          <p className="mt-6 text-sm">
            <a className="link" href={`${DOCS}/doctor.md`} target="_blank" rel="noreferrer">
              How detection and verification work
            </a>
          </p>
        </div>
      </div>
    </Section>
  );
}
