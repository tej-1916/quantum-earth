# Antigravity task: Quantum Earth frontend V1

You are working on the existing repository `tej-1916/quantum-earth`, branch `feat/scroll-frontend-v1`. Read the root `AGENTS.md` before making changes. Do not recreate the repository or overwrite the user's work. Do not modify `references/screenshot-to-code`, which is a reference submodule.

## Immediate blocker: browser is blank

The existing Vite production build succeeds, but the browser at localhost:4173 displays an empty dark page. Inspect the runtime error in browser devtools, not just the build output. There is a likely JSX runtime issue: `frontend/src/App.jsx` uses JSX but imports only named hooks from React, while `frontend/package.json` has no `@vitejs/plugin-react` and there is no React Vite configuration. Confirm the actual failure and fix it correctly, preferably with a compatible React plugin and `vite.config.js` using the automatic JSX runtime. Do not simply hide the error. Validate dev AND preview, not only `npm run build`. Preserve the existing functionality.

## Product brief

Create an original, polished dark cinematic scroll-experience website called **Quantum Earth**. It explores two model baselines for classifying satellite Earth imagery, and later our proposed quantum spectral-spatial model.

Sections:
1. Full-screen Earth/satellite hero with strong typography, subtle animated grid/orbit, and scroll cue.
2. Mission: satellite imagery contains information about water, crops, and cities.
3. Landscape cards for coastline, agricultural fields, and urban image.
4. Classical model vs existing-style hybrid quantum model, with clear data flows and original animations.
5. Honest comparison table, with not-measured and not-connected states until real models exist.
6. Interactive image lab: responsive upload/dropzone and local preview. No fake inference results.
7. Future research direction and footer.

Use React + Vite; refine typography, layout, accessibility, responsiveness and reduced-motion support. Prefer performant native CSS transitions and only add dependencies if necessary. The CSS visual fallbacks may remain until proper images are added. The user has selected five image assets from an earlier conversation, but they are NOT currently in the repository: Earth hero, coastline, farmland, urban, quantum hardware. Ask the user to attach or copy those files to `frontend/public/images/` if you cannot access them; do not invent local paths or use random imagery as ML inputs. Before public publishing, ensure reuse rights. Inspiration is for visual direction, not exact copying.

## Validation / handoff

- Run `npm install`, `npm run dev`, `npm run build`, and `npm run preview`.
- Verify that the rendered page is nonblank and no uncaught browser-console error appears.
- Exercise mobile layout, navigation, keyboard interaction, upload selection, drag-and-drop validation and image preview.
- Record exactly what passed and what is still pending; take a screenshot if your environment supports it.
- Keep changes on `feat/scroll-frontend-v1`; make a small, descriptive commit and push to existing PR #1 only after the checks. Do not merge the PR without user approval.
