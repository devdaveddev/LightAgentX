import { useState, type ReactNode } from "react";
import { motion } from "motion/react";
import { Check, Copy } from "lucide-react";

/** A numbered section, laid out like a chapter: number in the margin, serif heading, prose lead. */
export function Section({
  id,
  n,
  title,
  lead,
  children,
}: {
  id: string;
  n: string;
  title: ReactNode;
  lead?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section id={id} className="border-t border-line">
      <div className="mx-auto max-w-6xl px-4 py-20 sm:px-6 md:py-28">
        <header className="grid gap-x-10 gap-y-2 md:grid-cols-[6rem_1fr]">
          <p className="pt-2 font-mono text-sm text-faint">{n}</p>
          <div>
            <h2 className="text-3xl leading-tight font-medium text-balance sm:text-[2.6rem]">{title}</h2>
            {lead && <div className="mt-5 max-w-[64ch] space-y-4 text-[1.06rem] leading-relaxed text-pretty text-muted">{lead}</div>}
          </div>
        </header>
        <div className="mt-12 md:mt-14">{children}</div>
      </div>
    </section>
  );
}

export function CopyButton({ text, className = "" }: { text: string; className?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      aria-label={copied ? "Copied" : "Copy to clipboard"}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          setTimeout(() => setCopied(false), 1600);
        } catch {
          /* clipboard unavailable */
        }
      }}
      className={`inline-flex size-11 shrink-0 cursor-pointer items-center justify-center rounded-md text-code-dim transition-colors hover:bg-white/10 hover:text-code-fg md:size-9 ${className}`}
    >
      <motion.span key={copied ? "y" : "n"} initial={{ scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}>
        {copied ? <Check className="size-4 text-[#86efac]" /> : <Copy className="size-4" />}
      </motion.span>
    </button>
  );
}

const PY_KEYWORDS = new Set(
  "def class return import from as with if else elif for in while try except raise lambda None True False and or not is async await yield pass".split(" "),
);

/** Tiny tokenizer for Python / shell snippets. Good enough for a landing page, not a parser. */
function highlight(code: string, lang: "python" | "bash" | "text") {
  if (lang === "text") return [<span key="0">{code}</span>];
  const re =
    lang === "python"
      ? /(#[^\n]*)|("""[\s\S]*?"""|"[^"\n]*"|'[^'\n]*')|(@\w+)|(\b\d+(?:\.\d+)?\b)|(\b[A-Za-z_]\w*\b)/g
      : /(#[^\n]*)|("[^"\n]*"|'[^'\n]*')|(--?[\w-]+)|(\b\d+(?:\.\d+)?\b)|(\b[A-Za-z_][\w-]*\b)/g;
  const out: ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  let i = 0;
  while ((m = re.exec(code))) {
    if (m.index > last) out.push(code.slice(last, m.index));
    const [tok, com, str, dec, num, word] = m;
    let cls = "";
    if (com) cls = "tk-com";
    else if (str) cls = "tk-str";
    else if (dec) cls = lang === "python" ? "tk-dec" : "tk-kw";
    else if (num) cls = "tk-num";
    else if (word) {
      const next = code[m.index + tok.length];
      if (lang === "python" && PY_KEYWORDS.has(word)) cls = "tk-kw";
      else if (lang === "python" && /^[A-Z]/.test(word)) cls = "tk-cls";
      else if (lang === "python" && next === "(") cls = "tk-fn";
      else if (lang === "bash" && (m.index === 0 || code[m.index - 1] === "\n")) cls = "tk-fn";
    }
    out.push(cls ? <span key={i++} className={cls}>{tok}</span> : tok);
    last = m.index + tok.length;
  }
  if (last < code.length) out.push(code.slice(last));
  return out;
}

export function Panel({ title, action, children, className = "" }: { title: ReactNode; action?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <div className={`overflow-hidden rounded-lg border border-code-line bg-code text-code-fg ${className}`}>
      <div className="flex min-h-11 items-center justify-between border-b border-code-line pr-1 pl-4">
        <span className="truncate font-mono text-xs text-code-dim">{title}</span>
        {action}
      </div>
      {children}
    </div>
  );
}

export function CodeBlock({
  code,
  lang = "python",
  title,
  className = "",
}: {
  code: string;
  lang?: "python" | "bash" | "text";
  title?: string;
  className?: string;
}) {
  return (
    <Panel title={title ?? lang} action={<CopyButton text={code} />} className={className}>
      <pre className="overflow-x-auto p-4 font-mono text-[13px] leading-relaxed">
        <code>{highlight(code, lang)}</code>
      </pre>
    </Panel>
  );
}

export function GitHubIcon({ className = "size-4" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" className={className}>
      <path d="M12 .5C5.65.5.5 5.65.5 12a11.5 11.5 0 0 0 7.86 10.92c.58.1.79-.25.79-.56v-2c-3.2.7-3.87-1.37-3.87-1.37-.53-1.33-1.28-1.69-1.28-1.69-1.05-.71.08-.7.08-.7 1.15.08 1.76 1.19 1.76 1.19 1.03 1.76 2.7 1.25 3.36.96.1-.75.4-1.25.73-1.54-2.55-.29-5.24-1.28-5.24-5.68 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.41-2.69 5.38-5.25 5.67.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z" />
    </svg>
  );
}
