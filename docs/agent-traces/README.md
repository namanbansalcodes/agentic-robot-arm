# Coding-agent disclosure

This project was built with an AI coding agent, and that is disclosed here in full rather
than implied.

## Tooling

| | |
|---|---|
| Harness | Claude Code (CLI) |
| Model | `claude-opus-5` — every turn of every session |
| Sessions | 2: **session 1** (build: spike → primitives → agent → harness → the VoLo retarget), **session 2** (packaging and submission) |
| Subagents | 18 dispatches, all `general-purpose`, used for isolated, verifiable units of work |
| Skills used | `superpowers:writing-plans` (produced `implementation-plan.md`), `superpowers:subagent-driven-development` (executed it task-by-task), `claude-api` (Gemini model/pricing facts), `artifact-design` (the report and status pages) |

## Files here

| file | what it is |
|---|---|
| `implementation-plan.md` | the plan the whole build was executed against, written before any project code — a copy of `docs/superpowers/plans/2026-08-29-self-verifying-robot-arm.md` |
| `session-1-build.jsonl` | full raw transcript of the build session, including every failed attempt |
| `session-2-packaging.jsonl` | full raw transcript of the packaging session |

The transcripts are the unedited record. **The only modification is that any Google API
key shaped token is replaced with `GEMINI_API_KEY_REDACTED`.**

## What the agent actually did, and where it was wrong

The transcripts are long; `IMPROVEMENT_CHANGELOG.md` is the readable version, and it
deliberately keeps the rows where an earlier conclusion was overturned:

- **`retract()` dropped held blocks.** The agent tested it *once*, saw the block survive,
  and wrote down "risk cleared". That conclusion was wrong — a 27-run re-test found an
  ~11% drop rate. N=1 verification is exactly the open-loop confidence this project exists
  to measure, committed by its own author.
- **The firewall leaked twice** after being declared done, and was only fixed by attacking
  it with working exploits instead of confirming what it already caught.
- **The whole task was retargeted** after the user challenged whether the baseline was
  crippled. Decomposing the baseline's 17 false successes showed it scored 94% on the
  scenes that were fair and functional: single pick-and-place had no capability headroom,
  and what was about to be presented as agentic reasoning was substantially a perception
  gotcha. The user was right, and the eval was rebuilt around long-horizon rearrangement.

Every one of those was caught by re-measuring rather than by review, which is the same
argument the project makes about robots.
