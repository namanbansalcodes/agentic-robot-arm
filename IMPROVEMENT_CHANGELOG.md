# Improvement Changelog

Every meaningful iteration, with the evidence that drove the next decision.
Written as work happens, not reconstructed afterwards.

| Stage | What was tried and why | Evidence | Decision / learning |
|---|---|---|---|
| **Spike** | Prove panda-gym + PyBullet runs headless on CPU, that a hard-coded grasp lifts a block, and that we can render an RGB frame — before writing any project code. | `make spike`: headless sim up in 4.4 s wall; scripted grasp lifted block to `z=0.189` with finger width `0.049`; 480x360 RGB frame written. | Stack is viable, no fallback needed. Pinned pybullet 3.2.6 / panda-gym 3.0.7. |
| **Spike — macOS build** | pybullet 3.2.6 would not build on Apple Silicon. | `error: expected identifier or '('` at `_stdio.h:322`; the bundled zlib does `#define fdopen(fd,mode) NULL`, colliding with the macOS SDK declaration of `fdopen`. | Build with `CFLAGS="-std=gnu17 -Dfdopen=fdopen"`. Baked into `make setup`, documented in `REPRODUCTION.md`. |
| **Camera design** | Needed pixel to world unprojection owned by our code, to keep coordinates out of the VLM's hands. | Project/unproject round-trip on both cameras recovers test points to **< 3 mm**. | Own the view/projection matrices rather than using panda-gym's opaque `render()`. |
| **Camera placement** | First overhead camera was nadir over the table with the arm at its home pose. | Rendered frame: the arm occluded roughly a third of the table and a green cube was **completely invisible**. | `look()` retracts the arm to `(-0.28, 0, 0.42)` before imaging, as a real cell would. Occlusion eliminated. |
| **Scene contrast** | Default panda-gym palette is a light grey arm on a light grey table — low contrast for the human reading trajectory pages and for the L3 verifier. | Segmentation re-measured on the dark-table frame: identical 4 blobs, correct kinds, zero false positives (slate saturation sits below every mask's `S` floor). | Dark slate table on a light floor. Contrast improved at **no cost** to perception. |
| **Perception thresholds** | Planned to separate bowls from cubes by contour fill ratio. | Measured on a real frame: cube ~ **1,400 px**, bowl ~ **14,600 px**, but fill ratio is ~0.95 for *both* — from nadir a bowl reads as a filled square, not a ring. | Fill-ratio gate **removed**; classify on area alone, 10x margin. `fill_ratio` still recorded as evidence. |
| **VLM selection** | Needed a vision-language model suited to embodied reasoning, not a general chat model. | `gemini-robotics-er-2-preview` verified live: correct visual answer (`'yes'`; 1117 in / 1 out / 76 thought tokens) and a correct unprompted tool call `grasp(green_cube_1)` from a photo plus detection list. ~**$0.003 per call**. | Selected. Full 150-episode eval projects to ~$5 live; the free tier may cover it outright. |
| **Determinism** | The brief called for `temperature=0`. | The Gemini Interactions API exposes **`seed`, not `temperature`** — no temperature parameter exists on this surface. | Use `seed=0`, `thinking_level="low"`. Reproducibility of reported numbers comes from the replay cache. Stated plainly in the README rather than claiming a knob we do not have. |

<!-- Rows below are appended by the evidence pass (plan Task 14), each with numbers from an actual harness run. -->
