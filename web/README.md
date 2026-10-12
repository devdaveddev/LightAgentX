# LightAgentX website

Landing page for LightAgentX. Vite + React + TypeScript, Tailwind CSS v4, Framer Motion.

```bash
npm install
npm run dev       # http://localhost:5173
npm run build     # static site in dist/
```

Sections live in `src/components/`, one file each. Design tokens (colors, fonts) are in `src/index.css` under `@theme`.
Animations respect the visitor's reduced-motion setting (`MotionConfig reducedMotion="user"`).

## Docker

```bash
docker build -t lightagentx-web .
docker run -p 8080:8080 lightagentx-web   # http://localhost:8080
```

Multi-stage build: Node compiles the site, then a non-root nginx serves `dist/` on port 8080
(gzip, long-lived caching for fingerprinted `/assets`, health check at `/healthz`).

## Deploy (Render)

`render.yaml` at the repo root defines the service. It builds `web/Dockerfile` and redeploys on
pushes that touch `web/`.
