import { MotionConfig } from "motion/react";
import { Nav } from "./components/Nav";
import { Hero } from "./components/Hero";
import { Loop } from "./components/Loop";
import { Modules } from "./components/Modules";
import { Sandbox } from "./components/Sandbox";
import { Registry } from "./components/Registry";
import { Doctor } from "./components/Doctor";
import { Equivalence, Footer, Limits, Quickstart } from "./components/Closing";

export default function App() {
  return (
    <MotionConfig reducedMotion="user">
      <a href="#loop" className="sr-only z-[70] rounded-md bg-accent px-4 py-2 text-accent-ink focus:not-sr-only focus:fixed focus:top-3 focus:left-3">
        Skip to content
      </a>
      <Nav />
      <main>
        <Hero />
        <Loop />
        <Modules />
        <Sandbox />
        <Registry />
        <Doctor />
        <Equivalence />
        <Limits />
        <Quickstart />
      </main>
      <Footer />
    </MotionConfig>
  );
}
