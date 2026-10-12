import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["favicon.svg", "apple-touch-icon.png"],
      manifest: {
        id: "/",
        name: "LightAgentX",
        short_name: "LightAgentX",
        description: "A from-scratch agentic AI framework in pure Python. Agents, without the magic.",
        start_url: "/",
        scope: "/",
        display: "standalone",
        orientation: "portrait-primary",
        background_color: "#0f172a",
        theme_color: "#0f172a",
        categories: ["developer", "education", "productivity"],
        icons: [
          { src: "pwa-192x192.png", sizes: "192x192", type: "image/png" },
          { src: "pwa-512x512.png", sizes: "512x512", type: "image/png" },
          { src: "maskable-512x512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
        ],
        shortcuts: [
          { name: "Quickstart", url: "/#quickstart" },
          { name: "Sandbox", url: "/#sandbox" },
        ],
      },
      workbox: {
        // Precache the whole site (incl. fonts) so it opens offline once installed.
        globPatterns: ["**/*.{js,css,html,svg,png,woff2}"],
        // The page is English; skip font subsets it never renders.
        globIgnores: ["**/*-{cyrillic,cyrillic-ext,greek,greek-ext,vietnamese}-*.woff2"],
        navigateFallbackDenylist: [/^\/healthz/],
        cleanupOutdatedCaches: true,
      },
    }),
  ],
});
