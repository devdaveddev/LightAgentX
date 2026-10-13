import { MotionConfig, motion, useScroll, useSpring } from "motion/react";
import { Nav } from "./components/Nav";
import { Hero } from "./components/Hero";
import { Loop, Stats } from "./components/Loop";
import { Modules } from "./components/Modules";
import { Sandbox } from "./components/Sandbox";
import { Registry } from "./components/Registry";
import { Doctor } from "./components/Doctor";
import { Equivalence, Footer, Quickstart } from "./components/Closing";

function ScrollProgress() {
  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, { stiffness: 200, damping: 40, restDelta: 0.001 });
  return <motion.div className="fixed inset-x-0 top-0 z-[60] h-0.5 origin-left bg-accent" style={{ scaleX }} aria-hidden />;
}

export default function App() {
  return (
    <MotionConfig reducedMotion="user">
      <a href="#loop" className="sr-only z-[70] rounded-md bg-accent px-4 py-2 text-accent-ink focus:not-sr-only focus:fixed focus:top-3 focus:left-3">
        Skip to content
      </a>
      <ScrollProgress />
      <Nav />
      <main>
        <Hero />
        <Stats />
        <Loop />
        <Modules />
        <Sandbox />
        <Registry />
        <Doctor />
        <Equivalence />
        <Quickstart />
      </main>
      <Footer />
    </MotionConfig>
  );
}
