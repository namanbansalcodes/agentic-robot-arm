# Reproduction

Everything reported in `results/report.html` and `docs/evidence/` is reproducible from a
clean checkout **with no API key and no network**, because every VLM response the eval
depends on is committed in `cache/`.

## Tested platform

| | |
|---|---|
| OS | macOS 26.1 (build 25B78), Apple Silicon (`arm64`) |
| Python | CPython 3.11.6 |
| Simulator | pybullet 3.2.6 (DIRECT mode, `renderer="Tiny"`), panda-gym 3.0.7 |
| Model | `gemini-robotics-er-2-preview` via `google-genai` 2.20.0 |

Linux `x86_64` should work unchanged and does not need the macOS build fix below; it has
not been tested by us, and that is stated rather than implied.

## 1. Setup

```bash
make setup
```

That is `uv venv --python 3.11 .venv` followed by a pinned install. **On Apple Silicon it
also sets `CFLAGS="-std=gnu17 -Dfdopen=fdopen"`, and without that flag pybullet 3.2.6 does
not build at all**:

```
error: expected identifier or '('   ...   _stdio.h:322
```

pybullet's bundled zlib does `#define fdopen(fd,mode) NULL`, which collides with the macOS
SDK declaration of `fdopen`. Overriding the macro back to itself is what makes the wheel
compile. This is baked into `make setup`; it is documented here because a judge who
installs by hand will hit it.

### Two dependency files, on purpose

- `requirements.txt` — our **direct** dependencies, pinned. The readable statement of
  intent, and what `make setup` installs.
- `requirements.lock.txt` — the **exact** environment the reported numbers were produced
  on, transitives included (`uv pip freeze`).

If you get a different number than this repo reports, diffing your environment against the
lock file is the first thing to try:

```bash
VIRTUAL_ENV=.venv uv pip freeze | diff - requirements.lock.txt
```

## 2. Tests, including the firewall

```bash
make test
```

This is not a formality. `tests/test_firewall.py` is the structural enforcement of the
project's central claim — that the agent never sees ground truth — and it carries positive
controls, so it can actually fail.

## 3. The headline target — offline, free

```bash
make judge
```

Runs `make test`, then the **entire eval** (2 conditions × 9 scenes × 5 seeds = 90
episodes) against the committed replay cache, then regenerates the report.

- **No `GEMINI_API_KEY` required.** No network access required.
- **Cost: $0.00.** The dollar figures in the report are what the *recording* run cost.
- Outputs: `results/report.md`, `results/report.html`, `results/episodes.jsonl`,
  `results/trajectories/*.html`.

**Check `replay drift` in the report header. It must read `0`.** Anything else means a
cached response did not match its recorded prompt hash and the run is not reproducible.

`results/` is gitignored — it is generated, never hand-edited. The curated, committed copy
a reviewer can read without running anything lives in `docs/evidence/` (`make evidence`).

## 4. Optional: the live run

```bash
cp SECRETS.example SECRETS   # then put your key in it
make judge-live
```

Calls the real model and **rewrites the cache as it goes**, so a crashed run resumes
instead of paying twice. Only needed to verify the cache was honestly recorded.

Note on determinism: the Interactions API exposes **`seed`, not `temperature`**. We send
`seed=0, thinking_level="low"`. A live re-run is therefore *close* to, but not guaranteed
byte-identical with, the recorded one — which is exactly why the replay cache, not the
model, is what makes the reported numbers reproducible.

## 5. Smaller targets

```bash
make one-shot    # the blind open-loop condition only
make agentic     # the self-verifying condition only
make report      # rebuild the report from an existing results/episodes.jsonl
make evidence    # refresh the committed judge-facing pack in docs/evidence/
make spike       # the original feasibility proof: headless sim, scripted grasp, RGB frame
```

## Measured runtime and cost

| run | wall clock | cost | needs a key |
|---|---|---|---|
| `make test` | RUNTIME_TESTS | $0 | no |
| `make judge` (replay, 90 episodes) | RUNTIME_JUDGE | **$0** | **no** |
| the original recording run (live) | — | RECORDED_COST | yes |

Nearly all of the replay wall clock is PyBullet stepping physics and rasterising frames,
not model latency: replay re-executes every trajectory in the simulator for real and
re-scores it with the oracle. Only the *language* is cached.
