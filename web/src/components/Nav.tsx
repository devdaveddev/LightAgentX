import { useState } from "react";
import { AnimatePresence, motion, useMotionValueEvent, useScroll } from "motion/react";
import { Menu, X } from "lucide-react";
import { GitHubIcon } from "./ui";
import { REPO, ease } from "../lib";

const LINKS = [
  { href: "#loop", label: "Loop" },
  { href: "#modules", label: "Modules" },
  { href: "#sandbox", label: "Sandbox" },
  { href: "#registry", label: "Registry" },
  { href: "#doctor", label: "Doctor" },
  { href: "#quickstart", label: "Quickstart" },
];

export function Logo() {
  return (
    <a href="#top" className="flex items-center gap-2.5 font-semibold tracking-tight" aria-label="LightAgentX home">
      <img src="/favicon.svg" alt="" className="size-7" />
      <span>
        LightAgent<span className="text-accent">X</span>
      </span>
    </a>
  );
}

export function Nav() {
  const { scrollY } = useScroll();
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  useMotionValueEvent(scrollY, "change", (y) => setScrolled(y > 12));

  return (
    <header
      className={`fixed inset-x-0 top-0 z-50 transition-colors duration-300 ${
        scrolled || open ? "border-b border-line bg-bg/80 backdrop-blur-lg" : "border-b border-transparent"
      }`}
    >
      <nav className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6" aria-label="Main">
        <Logo />
        <ul className="hidden items-center gap-1 md:flex">
          {LINKS.map((l) => (
            <li key={l.href}>
              <a
                href={l.href}
                className="rounded-md px-3 py-2 text-sm text-muted transition-colors hover:text-fg"
              >
                {l.label}
              </a>
            </li>
          ))}
        </ul>
        <div className="flex items-center gap-2">
          <a
            href={REPO}
            target="_blank"
            rel="noreferrer"
            className="hidden items-center gap-2 rounded-lg border border-line-strong px-3 py-2 text-sm transition-colors hover:bg-white/5 sm:inline-flex"
          >
            <GitHubIcon /> GitHub
          </a>
          <button
            type="button"
            className="inline-flex size-11 cursor-pointer items-center justify-center rounded-lg text-muted hover:text-fg md:hidden"
            aria-label={open ? "Close menu" : "Open menu"}
            aria-expanded={open}
            onClick={() => setOpen((o) => !o)}
          >
            {open ? <X className="size-5" /> : <Menu className="size-5" />}
          </button>
        </div>
      </nav>
      <AnimatePresence>
        {open && (
          <motion.ul
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0, transition: { duration: 0.2 } }}
            transition={{ duration: 0.3, ease }}
            className="overflow-hidden px-4 md:hidden"
          >
            {[...LINKS, { href: REPO, label: "GitHub" }].map((l) => (
              <li key={l.href}>
                <a
                  href={l.href}
                  onClick={() => setOpen(false)}
                  className="block border-t border-line py-3.5 text-muted hover:text-fg"
                >
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
