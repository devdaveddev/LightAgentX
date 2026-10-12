import { useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Menu, X } from "lucide-react";
import { REPO, ease } from "../lib";

const LINKS = [
  { href: "#loop", label: "Loop" },
  { href: "#modules", label: "Modules" },
  { href: "#sandbox", label: "Sandbox" },
  { href: "#registry", label: "Registry" },
  { href: "#doctor", label: "Doctor" },
  { href: "#quickstart", label: "Quickstart" },
];

export function Wordmark() {
  return (
    <a href="#top" className="font-serif text-xl font-medium tracking-tight" aria-label="LightAgentX, back to top">
      LightAgent<span className="text-accent">X</span>
    </a>
  );
}

export function Nav() {
  const [open, setOpen] = useState(false);
  return (
    <header className="fixed inset-x-0 top-0 z-50 border-b border-line bg-bg/90 pt-[env(safe-area-inset-top)] pr-[env(safe-area-inset-right)] pl-[env(safe-area-inset-left)] backdrop-blur">
      <nav className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4 sm:px-6" aria-label="Main">
        <Wordmark />
        <ul className="hidden items-center gap-6 md:flex">
          {LINKS.map((l) => (
            <li key={l.href}>
              <a href={l.href} className="text-sm text-muted transition-colors hover:text-fg">
                {l.label}
              </a>
            </li>
          ))}
          <li>
            <a href={REPO} target="_blank" rel="noreferrer" className="text-sm text-muted transition-colors hover:text-fg">
              GitHub ↗
            </a>
          </li>
        </ul>
        <button
          type="button"
          className="-mr-2 inline-flex size-11 cursor-pointer items-center justify-center rounded-md text-muted hover:text-fg md:hidden"
          aria-label={open ? "Close menu" : "Open menu"}
          aria-expanded={open}
          onClick={() => setOpen((o) => !o)}
        >
          {open ? <X className="size-5" /> : <Menu className="size-5" />}
        </button>
      </nav>
      <AnimatePresence>
        {open && (
          <motion.ul
            initial={{ height: 0 }}
            animate={{ height: "auto" }}
            exit={{ height: 0, transition: { duration: 0.18 } }}
            transition={{ duration: 0.25, ease }}
            className="overflow-hidden px-4 md:hidden"
          >
            {[...LINKS, { href: REPO, label: "GitHub ↗" }].map((l) => (
              <li key={l.href}>
                <a href={l.href} onClick={() => setOpen(false)} className="block border-t border-line py-3.5 text-muted hover:text-fg">
                  {l.label}
                </a>
              </li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </header>
  );
}
