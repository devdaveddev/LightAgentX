import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { VitePWA } from "vite-plugin-pwa";

// "/" for Docker/Render; GitHub Pages serves from /<repo>/ and sets BASE_PATH.
export default defineConfig({
  base: process.env.BASE_PATH ?? "/",
  plugins: [
    react(),
    tailwindcss(),
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["favicon.svg", "apple-touch-icon.png"],
      manifest: {
        id: "./",
        name: "LightAgentX",
        short_name: "LightAgentX",
        description: "An agent framework written from scratch in plain Python.",
        start_url: "./",
        scope: "./",
        display: "standalone",
        orientation: "portrait-primary",
        background_color: "#f6f4ee",
        theme_color: "#f6f4ee",
        categories: ["developer", "education", "productivity"],
        icons: [
          { src: "pwa-192x192.png", sizes: "192x192", type: "image/png" },
          { src: "pwa-512x512.png", sizes: "512x512", type: "image/png" },
          { src: "maskable-512x512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
        ],
        shortcuts: [
          { name: "Quickstart", url: "./#quickstart" },
          { name: "Sandbox", url: "./#sandbox" },
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
