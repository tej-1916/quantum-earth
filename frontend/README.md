# Quantum Earth frontend

An original, responsive scroll-experience website for comparing classical and quantum Earth-observation models.

## Run locally

Requires Node.js 20.19+ or 22.12+.

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite.

## Build

```bash
npm run build
npm run preview
```

## Current state

- Cinematic sections, responsive navigation and scroll effects are functional.
- Image upload supports local preview, file-type validation and a 10 MB limit.
- No inference endpoint, accuracy figures or predictions are simulated.
- Both model cards explicitly show "Not connected" until real backends exist.
- Optional owned/licensed visual assets may be added to `public/images/`. CSS artwork works in their absence.

Next milestone: select a real, appropriately licensed Earth-observation dataset and implement matched classical and hybrid quantum baselines behind an API.
