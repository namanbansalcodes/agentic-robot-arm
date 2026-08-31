# Does an agentic loop beat a one-shot plan on a robot arm?

A Franka Panda in simulation, driven by **Gemini Robotics-ER 2**. Two conditions that are
identical in every respect except one: whether the robot looks at the result of its own
actions before deciding the next one.

This replicates the central claim of **[VoLo: A Physical Orchestrator for Open-Vocabulary
Long-Horizon Manipulation](https://arxiv.org/abs/2606.07723)** (NVIDIA, June 2026) in a
controlled miniature. VoLo's baseline family is literally *"single action model (no
orchestrator)"* — systems lacking a monitoring and recovery loop. That is exactly our
`one_shot` condition, and VoLo runs on a Franka FR3 where we simulate a Franka Panda.

> **We do not reproduce VoLo.** That needs their benchmark and their hardware. We test the
> same hypothesis where every variable but one is held fixed.

---

## Who this is for, and what breaks today

A robotics engineer prototyping a VLM-driven arm. The model is good enough to plan a task
from a photo — so the tempting architecture is one call, full plan, execute. That works
until the world stops matching the plan, and then it fails **silently**: the arm finishes
its script and reports success while the blocks sit on the table. Nothing downstream knows.

The bottleneck is not planning quality. It is that a plan written at t=0 cannot see t=5.

---

## Results

Three scenarios, five seeds each, one episode per condition per seed.

| scenario | what it demands | one-shot | agentic | false successes |
|---|---|---|---|---|
| `disturb_h3` | Recovery from a mid-task disturbance | **0%** | **100%** | 5 → 0 |
| `mem_swap` | Memory / state tracking (VoLo *Swap*) | **20%** | **100%** | 4 → 0 |
| `match3` | Colour-matched routing | **60%** | **80%** | 2 → 1 |

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 230" width="100%" style="max-width:720px" role="img" aria-label="one-shot versus agentic success rate">
<g font-family="system-ui,sans-serif" font-size="12">
<rect x="150" y="8" width="11" height="11" rx="2" fill="#c0392b"/><text x="167" y="18" fill="currentColor">one-shot</text>
<rect x="245" y="8" width="11" height="11" rx="2" fill="#12803c"/><text x="262" y="18" fill="currentColor">agentic</text></g>
<text x="140" y="49" text-anchor="end" font-family="ui-monospace,monospace" font-size="11.5" fill="currentColor">disturb_h3</text>
<rect x="150" y="36" width="1.5" height="15" rx="2" fill="#c0392b"/><text x="157.5" y="48" font-size="11" font-family="ui-monospace,monospace" fill="currentColor">0%</text>
<rect x="150" y="54" width="500.0" height="15" rx="2" fill="#12803c"/><text x="656.0" y="66" font-size="11" font-family="ui-monospace,monospace" fill="currentColor">100%</text>
<text x="140" y="93" text-anchor="end" font-family="ui-monospace,monospace" font-size="11.5" fill="currentColor">mem_swap</text>
<rect x="150" y="80" width="100.0" height="15" rx="2" fill="#c0392b"/><text x="256.0" y="92" font-size="11" font-family="ui-monospace,monospace" fill="currentColor">20%</text>
<rect x="150" y="98" width="500.0" height="15" rx="2" fill="#12803c"/><text x="656.0" y="110" font-size="11" font-family="ui-monospace,monospace" fill="currentColor">100%</text>
<text x="140" y="137" text-anchor="end" font-family="ui-monospace,monospace" font-size="11.5" fill="currentColor">match3</text>
<rect x="150" y="124" width="300.0" height="15" rx="2" fill="#c0392b"/><text x="456.0" y="136" font-size="11" font-family="ui-monospace,monospace" fill="currentColor">60%</text>
<rect x="150" y="142" width="400.0" height="15" rx="2" fill="#12803c"/><text x="556.0" y="154" font-size="11" font-family="ui-monospace,monospace" fill="currentColor">80%</text>
<line x1="150" y1="30" x2="150" y2="160" stroke="currentColor" stroke-opacity=".35"/>
</svg>

**A "false success" is an episode where the robot claimed success the ground-truth oracle
scored as failure.** It is the failure mode that matters: a robot that fails and says so is
recoverable; one that fails and reports success is not.

### Success decays as the horizon grows

One-shot only, same model and same scene family, varying only how many blocks must be moved:

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 220" width="100%" style="max-width:560px" role="img" aria-label="one-shot success versus horizon length">
<line x1="60" y1="180.0" x2="540" y2="180.0" stroke="currentColor" stroke-opacity=".12"/>
<text x="52" y="184.0" text-anchor="end" font-size="10.5" font-family="ui-monospace,monospace" fill="currentColor" fill-opacity=".7">0%</text>
<line x1="60" y1="140.0" x2="540" y2="140.0" stroke="currentColor" stroke-opacity=".12"/>
<text x="52" y="144.0" text-anchor="end" font-size="10.5" font-family="ui-monospace,monospace" fill="currentColor" fill-opacity=".7">25%</text>
<line x1="60" y1="100.0" x2="540" y2="100.0" stroke="currentColor" stroke-opacity=".12"/>
<text x="52" y="104.0" text-anchor="end" font-size="10.5" font-family="ui-monospace,monospace" fill="currentColor" fill-opacity=".7">50%</text>
<line x1="60" y1="60.0" x2="540" y2="60.0" stroke="currentColor" stroke-opacity=".12"/>
<text x="52" y="64.0" text-anchor="end" font-size="10.5" font-family="ui-monospace,monospace" fill="currentColor" fill-opacity=".7">75%</text>
<line x1="60" y1="20.0" x2="540" y2="20.0" stroke="currentColor" stroke-opacity=".12"/>
<text x="52" y="24.0" text-anchor="end" font-size="10.5" font-family="ui-monospace,monospace" fill="currentColor" fill-opacity=".7">100%</text>
<path d="M 60.0 20.0 L 220.0 20.0 L 380.0 52.0 L 540.0 148.0" fill="none" stroke="#c0392b" stroke-width="2.5"/>
<circle cx="60.0" cy="20.0" r="4.5" fill="#c0392b"/>
<text x="60.0" y="9.0" text-anchor="middle" font-size="11" font-family="ui-monospace,monospace" fill="currentColor">100%</text>
<text x="60.0" y="206" text-anchor="middle" font-size="11" font-family="system-ui,sans-serif" fill="currentColor" fill-opacity=".75">1 block</text>
<circle cx="220.0" cy="20.0" r="4.5" fill="#c0392b"/>
<text x="220.0" y="9.0" text-anchor="middle" font-size="11" font-family="ui-monospace,monospace" fill="currentColor">100%</text>
<text x="220.0" y="206" text-anchor="middle" font-size="11" font-family="system-ui,sans-serif" fill="currentColor" fill-opacity=".75">2 blocks</text>
<circle cx="380.0" cy="52.0" r="4.5" fill="#c0392b"/>
<text x="380.0" y="41.0" text-anchor="middle" font-size="11" font-family="ui-monospace,monospace" fill="currentColor">80%</text>
<text x="380.0" y="206" text-anchor="middle" font-size="11" font-family="system-ui,sans-serif" fill="currentColor" fill-opacity=".75">3 blocks</text>
<circle cx="540.0" cy="148.0" r="4.5" fill="#c0392b"/>
<text x="540.0" y="137.0" text-anchor="middle" font-size="11" font-family="ui-monospace,monospace" fill="currentColor">20%</text>
<text x="540.0" y="206" text-anchor="middle" font-size="11" font-family="system-ui,sans-serif" fill="currentColor" fill-opacity=".75">4 blocks</text>
</svg>

Nothing about the model changed between these points. The only variable is how many steps it
had to commit to before seeing any result.

---

## What existed before, and what we built

**Existed:** [panda-gym](https://github.com/qgallouedec/panda-gym) and PyBullet, the Franka
Panda URDF, the Gemini API and `gemini-robotics-er-2-preview`, OpenCV.

**Built here:** the five primitives the model drives, the perception stack, the `RobotIO`
blindfold, the ground-truth oracle and its firewall, both agent loops (one-shot and ReAct
with memory), the disturbance mechanism, the nine scenes, the eval harness, the replay
cache, and the report and trajectory generators. 237 tests.

## The firewall: the agent never sees ground truth

The simulator knows where everything is. The scoring harness may read that; **the agent may
never**. Enforced structurally, not by convention:

| layer | blocks |
|---|---|
| import scan | `robotsim.oracle`, `robotsim.world`, `harness` |
| attribute scan | `.oracle`, `.world`, `.sim`, `._bowl_centers`, `get_base_*`, `physics_client*` |
| dynamic scan | `__import__`, `eval`, `exec`, `importlib` |
| call-site scan | ground truth injected as a parameter (`def act(self, oracle)`) |
| `RobotIO` surface | exactly 11 methods, none returning the pose of anything the robot is not holding |

It was attacked five times during development and **leaked twice** before it held. Both
leaks are documented in `IMPROVEMENT_CHANGELOG.md`. Every scan carries positive controls, so
a detector that stopped detecting would fail the build rather than pass silently.

## Fairness by construction

Both conditions share the same model, the same tool schema, the same dispatcher, and the
same **1,430-byte prompt preamble — the same Python object, verified byte-identical, first
divergence at character 1435.** A test fails the build if they ever drift apart. The only
difference is the loop.

---

## Reproducing

```bash
make setup     # pinned deps; on macOS builds pybullet with a required CFLAGS override
make test      # 237 tests
make judge     # replays every episode offline from the committed cache — free, no API key
```

`make judge` needs no `GEMINI_API_KEY`. Live runs (`make judge-live`) cost roughly $0.01 per
one-shot episode and $0.05–0.11 per agentic episode. Full details in `REPRODUCTION.md`.

---

## The main failure mode, and the hot take

The most instructive bug in this project was ours, not the model's. The agentic condition
scored **0/5** on the disturbance scenario and blamed an unreachable block. The scene checked
out — the block was inside the workspace and detected. The trace showed the agent grasping a
second block *while still holding the first*, which dragged the held one out of reach. Every
later attempt then returned a genuine `unreachable`. **Our primitive was corrupting the world,
and the result read as an impossible task.** The agent had been trying to recover the whole
time. One guard later: **0% → 100%**.

**Hot take:** the interesting number is not success rate, it is the gap between what a robot
claims and what it did. A blind planner's success rate degrades gracefully as tasks get
longer — but its *honesty* does not degrade at all, because it was never honest; it simply
reports success at a fixed rate regardless of what happened. Closed-loop execution buys
capability, and that is worth measuring. What it really buys is a robot whose report you can
act on. Judge embodied agents on the honesty gap, not the leaderboard.
