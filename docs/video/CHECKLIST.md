# Before you hit record

Work top to bottom. Every item is something the script says on camera, so if one fails,
the script changes — not the take.

## 1. The build is green from a clean checkout

```bash
make test        # must be all-green; there is no acceptable "known failure"
```

Two tests referenced `h4_quad` after it was dropped from `scenes.yaml`, which made
`make judge` exit non-zero — retargeted onto `h3_triple` and `match3`. If you touch the
scene set again, grep for the removed id before recording:

```bash
grep -rn "h4_quad" tests/ harness/ agent/ primitives/ robotsim/
```

## 2. The headline run exists and is the one on screen

```bash
make judge
```

- `replay drift` in the report header must read **0**. The script says this out loud.
- `results/report.html` must be regenerated *after* the last code change. A stale report
  behind a fresh terminal is the one thing in this video that would be dishonest.

## 3. The evidence pack is committed

```bash
make evidence     # writes docs/evidence/
```

`README.md` links to `docs/evidence/`. If that directory is missing, every judge who
clicks it gets a 404 before they have run anything.

## 4. `REPRODUCTION.md` has no placeholders left

`RUNTIME_TESTS`, `RUNTIME_JUDGE`, `RECORDED_COST` must all be real measured values.
Reproducibility is 15 points and this is the page that earns them.

## 5. The episode you walk through in Segment 4 actually contains the beats

The script narrates: a grasp that closes on air, an L2 catch at `0.0000`, a retry at
`~0.044`, an L3 photo check, the mid-episode disturbance, and an honest `report_done`.

Open the trajectory page and confirm each one is really there:

```bash
open results/trajectories/agentic_disturb_h3_s0.html
```

If a beat is missing at seed 0, check the other seeds and swap the episode — do **not**
narrate a beat the page does not show.

```bash
ls results/trajectories/ | grep disturb
```

## 6. The GIFs match what the report now says

`docs/videos/*.gif` were recorded before the `grasp()` already-holding fix, which moved
`disturb_h3` agentic from 0/5 to 100%. Check that the outcome captioned under each GIF in
`README.md` still agrees with the per-scene table in `results/report.md`. If a GIF now
disagrees with the eval, either re-record it or relabel it in the README as the earlier
recorded episode it is — a judge who spots the mismatch unaided will read it as spin.

## 7. Screen setup

- 1920×1080 or better, terminal at 18pt+, dark theme.
- Browser zoom on `report.html` and the trajectory pages set so the numbers are legible
  at 720p — assume the judge watches in a window, not full screen.
- Hide notifications, hide the dock, clear the shell of prior scrollback.
- Have these open in tabs, in script order, before you start:
  1. `docs/videos/one_shot_small.gif`
  2. `agent/baseline.py`
  3. `results/trajectories/agentic_disturb_h3_s0.html`
  4. `tests/test_firewall.py`
  5. `results/report.html`
  6. `IMPROVEMENT_CHANGELOG.md`

## 8. Timing

Read the script aloud once with a stopwatch before recording. The hard cap is 5:00 and it
is enforced. If you are over, use the trim list at the bottom of `SCRIPT.md` rather than
speeding up — the numbers need to be audible.
