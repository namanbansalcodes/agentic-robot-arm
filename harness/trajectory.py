"""One self-contained HTML page per episode: every step, every photo, every verdict.

This is the page a reader opens when a table row surprises them. The headline number
in the report is an aggregate and an aggregate can only ever be believed; a trajectory
page is the receipt. It shows the frame the model was looking at, the words it wrote
about that frame, the raw feedback the primitive handed back, and which verification
layer -- if any -- objected. A false success is legible here in a way it can never be
in a rate: the claim is printed next to the photo that contradicts it.

Three constraints shape the implementation:

  * NO network assets. No CDN, no webfont, no JS library. The `results/` folder has to
    survive being zipped, mailed, and opened on a laptop with no internet, because
    that is how evidence actually gets read. Everything is one inline <style>.
  * RELATIVE image paths. `feedback.image_path` is an absolute-ish path from the run
    that produced it; baking it into the page would break the moment the folder moved.
    Only the basename survives, under `image_rel_prefix`.
  * BOTH themes. Colours are CSS custom properties with a `prefers-color-scheme`
    override. A hardcoded `#000` reads as invisible on the ground half the audience
    is using, and the aperture colour-coding is the point of the page.
"""
from __future__ import annotations

import html
import pathlib
from typing import Optional

# The L2 cliff, quoted rather than reinvented. This is primitives.api's
# EMPTY_GRIP_THRESHOLD -- the same number the L2 layer itself compares against, so a
# cell this page paints "closed on air" is exactly a cell L2 would have objected to.
# tests/test_report.py asserts the two stay equal, so drift is a test failure, not a
# silently wrong colour.
CLOSED_ON_AIR_M = 0.012

# The other two bands are read off the same measurements primitives/api.py records in
# its own comment: closed-on-air ~0.000, a held cube ~0.044, open-and-empty ~0.080.
# HOLDING_M's upper edge is harness/episode.py's HELD_APERTURE_M. These are display
# bands, not decision thresholds -- nothing in the eval branches on them.
HOLDING_M = (0.03, 0.06)
OPEN_EMPTY_M = 0.07

APERTURE_LABELS = {
    "closed": "closed on air",
    "holding": "holding",
    "open": "open / empty",
    "between": "between bands",
}


def aperture_class(width: float) -> str:
    """Which L2 band an aperture falls in. Purely for colour and a one-word label."""
    if width < CLOSED_ON_AIR_M:
        return "closed"
    if HOLDING_M[0] <= width <= HOLDING_M[1]:
        return "holding"
    if width > OPEN_EMPTY_M:
        return "open"
    return "between"


def step_dicts(result) -> Optional[list]:
    """The per-step trace, if this results record actually carries one.

    Records written after the schema fix carry `trace_steps`, the full per-step list.
    Older records stored only `len(trace.steps)` in an int `steps` field, so a page
    built from one gets a count and nothing else. Rather than crash or, worse, render
    a plausible-looking empty trajectory, this returns None and the page says out loud
    that the trace is not in the file.
    """
    # Preferred source: the dedicated trace field added to the persisted schema.
    trace = getattr(result, "trace_steps", None)
    if isinstance(trace, list) and trace:
        return trace
    # Fallback: a live EpisodeResult whose `steps` is still the list itself.
    steps = getattr(result, "steps", None)
    if isinstance(steps, list):
        return steps
    return None


def _esc(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _image_src(image_path, prefix: str) -> Optional[str]:
    if not image_path:
        return None
    name = pathlib.Path(str(image_path)).name
    if not name:
        return None
    return f"{prefix.rstrip('/')}/{name}" if prefix else name


def _verdict_badge(verdict) -> str:
    """Which layer objected, or that nothing did.

    A missing verdict and a passing verdict are DIFFERENT facts and are drawn
    differently: "not checked" means the condition had that layer switched off, and
    collapsing it into "passed" would make an ablation row look like verification it
    never ran.
    """
    if not verdict:
        return '<span class="badge verdict-none">not checked</span>'
    layer = verdict.get("layer")
    reason = verdict.get("reason") or ""
    if verdict.get("informational"):
        return (f'<span class="badge verdict-info">{_esc(layer or "L3")} '
                f'informational</span>'
                f'<div class="verdict-reason">{_esc(reason)}</div>')
    if verdict.get("ok"):
        label = f"{layer} passed" if layer else "passed"
        body = f'<div class="verdict-reason">{_esc(reason)}</div>' if reason else ""
        return f'<span class="badge verdict-pass">{_esc(label)}</span>{body}'
    return (f'<span class="badge verdict-fail">{_esc(layer or "?")} objected</span>'
            f'<div class="verdict-reason">{_esc(reason)}</div>')


def _feedback_block(feedback: dict) -> str:
    """The raw evidence, printed. Every verification layer reads this object, so the
    page prints the whole of it rather than a summary -- a reader checking whether L2
    *could* have caught something needs the aperture, not a paraphrase of it."""
    lines = [f"status        : {feedback.get('status', '?')}"]
    if feedback.get("error"):
        lines.append(f"error         : {feedback['error']}")
    width = feedback.get("fingers_width")
    if isinstance(width, (int, float)):
        band = aperture_class(float(width))
        lines.append(f"aperture (m)  : {float(width):.4f}   [{APERTURE_LABELS[band]}]")
    ee = feedback.get("ee_position")
    if ee:
        try:
            lines.append("ee_position   : ["
                         + ", ".join(f"{float(v):.3f}" for v in ee) + "]")
        except (TypeError, ValueError):
            lines.append(f"ee_position   : {ee}")
    detections = feedback.get("detections") or []
    if detections:
        seen = ", ".join(f"{d.get('id')} ({d.get('where')})" for d in detections)
        lines.append(f"visible ids   : {seen}")
    else:
        lines.append("visible ids   : nothing detected")
    lines.append(f"sim_steps     : {feedback.get('sim_steps', 0)}")
    if feedback.get("note"):
        lines.append(f"note          : {feedback['note']}")
    return _esc("\n".join(lines))


def _aperture_chip(feedback: dict) -> str:
    width = feedback.get("fingers_width")
    if not isinstance(width, (int, float)):
        return ""
    band = aperture_class(float(width))
    return (f'<span class="chip aperture aperture-{band}">'
            f'{float(width):.4f} m &middot; {APERTURE_LABELS[band]}</span>')


def _step_row(index: int, step: dict, prefix: str) -> str:
    feedback = step.get("feedback") or {}
    src = _image_src(feedback.get("image_path"), prefix)
    if src:
        thumb = (f'<img src="{_esc(src)}" alt="frame after step {index}" '
                 f'loading="lazy">')
    else:
        thumb = '<div class="no-frame">no frame</div>'
    args = step.get("args") or {}
    args_text = ", ".join(f"{k}={v!r}" for k, v in args.items()) or "no arguments"
    reasoning = step.get("reasoning") or ""
    reasoning_block = (f'<blockquote class="reasoning">{_esc(reasoning)}</blockquote>'
                       if reasoning.strip() else
                       '<p class="muted">no reasoning text on this step</p>')
    return f"""    <li class="step">
      <div class="thumb">{thumb}<div class="step-no">step {index}</div></div>
      <div class="detail">
        <div class="call"><code>{_esc(step.get('primitive'))}</code>
          <span class="args">{_esc(args_text)}</span>{_aperture_chip(feedback)}</div>
        {reasoning_block}
        <pre class="feedback">{_feedback_block(feedback)}</pre>
        <div class="verdict">{_verdict_badge(step.get('verdict'))}</div>
      </div>
    </li>"""


def _outcome_chips(result) -> str:
    claimed = "success" if result.claimed_success else "failure"
    actual = "success" if result.actual_success else "failure"
    return f"""    <div class="outcomes">
      <div class="outcome claim-{claimed}">
        <div class="outcome-label">agent claimed</div>
        <div class="outcome-value">{claimed}</div>
        <div class="outcome-note">{_esc(result.claim_reason or "no reason given")}</div>
      </div>
      <div class="outcome truth-{actual}">
        <div class="outcome-label">oracle measured</div>
        <div class="outcome-value">{actual}</div>
        <div class="outcome-note">ground truth, read after the scene settled</div>
      </div>
    </div>"""


CSS = """:root {
  color-scheme: light dark;
  --bg: #f7f7f8; --panel: #ffffff; --ink: #16181d; --muted: #5c6270;
  --line: #d8dbe2; --code-bg: #f0f1f4;
  --good: #12694a; --good-bg: #dbf0e5;
  --bad: #a01722; --bad-bg: #fbdfe1;
  --warn: #7a5200; --warn-bg: #fbeecd;
  --info: #1f4f8f; --info-bg: #dee9f8;
  --neutral: #4a4f5a; --neutral-bg: #e8eaee;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14161a; --panel: #1c1f25; --ink: #e8eaf0; --muted: #9aa2b1;
    --line: #333842; --code-bg: #23262d;
    --good: #6fd6a6; --good-bg: #123527;
    --bad: #ff9aa2; --bad-bg: #3d1519;
    --warn: #f0c264; --warn-bg: #3a2c0c;
    --info: #8fc0f5; --info-bg: #142740;
    --neutral: #b3bac6; --neutral-bg: #262a31;
  }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 1.6rem clamp(0.8rem, 4vw, 3rem); background: var(--bg);
  color: var(--ink); font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI",
  Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 1080px; margin: 0 auto; }
a { color: var(--info); }
h1 { font-size: 1.35rem; margin: 0 0 0.2rem; }
h2 { font-size: 1.05rem; margin: 2rem 0 0.6rem; }
.muted { color: var(--muted); }
.back { display: inline-block; margin-bottom: 1rem; font-size: 0.85rem; }
.false-success-banner { background: var(--bad-bg); color: var(--bad);
  border: 2px solid currentColor; border-radius: 8px; padding: 0.8rem 1rem;
  font-weight: 700; letter-spacing: 0.01em; margin: 0 0 1.1rem; }
.false-success-banner small { display: block; font-weight: 400; margin-top: 0.25rem;
  letter-spacing: 0; }
.meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 0.55rem; background: var(--panel); border: 1px solid var(--line);
  border-radius: 8px; padding: 0.9rem 1rem; margin-bottom: 0.9rem; }
.meta div { min-width: 0; }
.meta dt, .meta .k { color: var(--muted); font-size: 0.72rem;
  text-transform: uppercase; letter-spacing: 0.06em; }
.meta .v { font-weight: 600; overflow-wrap: anywhere; }
.outcomes { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 0.75rem; margin-bottom: 1.4rem; }
.outcome { border: 1px solid var(--line); border-left-width: 5px; border-radius: 8px;
  padding: 0.75rem 0.9rem; background: var(--panel); }
.outcome-label { font-size: 0.72rem; text-transform: uppercase; color: var(--muted);
  letter-spacing: 0.06em; }
.outcome-value { font-size: 1.25rem; font-weight: 700; text-transform: capitalize; }
.outcome-note { font-size: 0.82rem; color: var(--muted); margin-top: 0.2rem; }
.claim-success, .truth-success { border-left-color: var(--good); }
.claim-success .outcome-value, .truth-success .outcome-value { color: var(--good); }
.claim-failure, .truth-failure { border-left-color: var(--bad); }
.claim-failure .outcome-value, .truth-failure .outcome-value { color: var(--bad); }
ol.steps { list-style: none; margin: 0; padding: 0; }
.step { display: grid; grid-template-columns: 200px minmax(0, 1fr); gap: 1rem;
  background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 0.9rem; margin-bottom: 0.8rem; }
@media (max-width: 640px) { .step { grid-template-columns: 1fr; } }
.thumb img { width: 100%; height: auto; border-radius: 6px;
  border: 1px solid var(--line); display: block; }
.no-frame { display: flex; align-items: center; justify-content: center;
  aspect-ratio: 1 / 1; border: 1px dashed var(--line); border-radius: 6px;
  color: var(--muted); font-size: 0.8rem; }
.step-no { margin-top: 0.35rem; font-size: 0.72rem; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.06em; }
.call { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem;
  margin-bottom: 0.5rem; }
.call code { background: var(--code-bg); padding: 0.1rem 0.4rem; border-radius: 4px;
  font-weight: 700; }
.args { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo,
  monospace; font-size: 0.82rem; overflow-wrap: anywhere; }
blockquote.reasoning { margin: 0 0 0.55rem; padding: 0.4rem 0 0.4rem 0.8rem;
  border-left: 3px solid var(--line); color: var(--ink); white-space: pre-wrap; }
pre.feedback, pre { background: var(--code-bg); border: 1px solid var(--line);
  border-radius: 6px; padding: 0.6rem 0.75rem; overflow-x: auto; margin: 0 0 0.55rem;
  font: 12.5px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }
.chip, .badge { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 999px;
  font-size: 0.76rem; font-weight: 600; border: 1px solid currentColor; }
.aperture-closed { color: var(--bad); background: var(--bad-bg); }
.aperture-holding { color: var(--good); background: var(--good-bg); }
.aperture-open { color: var(--warn); background: var(--warn-bg); }
.aperture-between { color: var(--neutral); background: var(--neutral-bg); }
.verdict-pass { color: var(--good); background: var(--good-bg); }
.verdict-fail { color: var(--bad); background: var(--bad-bg); }
.verdict-info { color: var(--info); background: var(--info-bg); }
.verdict-none { color: var(--neutral); background: var(--neutral-bg); }
.verdict-reason { margin-top: 0.3rem; font-size: 0.84rem; color: var(--muted);
  overflow-wrap: anywhere; }
.notice { background: var(--warn-bg); color: var(--warn); border: 1px solid
  currentColor; border-radius: 8px; padding: 0.8rem 1rem; }
footer { margin-top: 2rem; font-size: 0.8rem; color: var(--muted);
  border-top: 1px solid var(--line); padding-top: 0.7rem; }
"""


def _page(title: str, body: str) -> str:
    return (f"<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            f"<meta charset=\"utf-8\">\n"
            f"<meta name=\"viewport\" content=\"width=device-width, "
            f"initial-scale=1\">\n<title>{_esc(title)}</title>\n"
            f"<style>\n{CSS}</style>\n</head>\n<body>\n<main>\n{body}\n</main>\n"
            f"</body>\n</html>\n")


def _meta_grid(result, step_count) -> str:
    cells = [
        ("condition", result.condition),
        ("scene", result.scene_id),
        ("seed", result.seed),
        ("failure mode", result.failure_mode),
        ("steps", step_count),
        ("recoveries", result.recoveries),
        ("escalated to human", "yes" if result.asked_human else "no"),
        ("vlm calls", result.vlm_calls),
        ("l3 checks", result.l3_calls),
        ("cost", f"${result.cost_usd:.4f}"),
        ("wall seconds", f"{result.wall_seconds:.1f}"),
        ("stop reason", result.stop_reason or "not recorded"),
    ]
    return ('  <div class="meta">\n'
            + "\n".join(f'    <div><div class="k">{_esc(k)}</div>'
                        f'<div class="v">{_esc(v)}</div></div>' for k, v in cells)
            + "\n  </div>")


def render_trajectory(result, out_dir, image_rel_prefix: str = "images"):
    """Write one episode's page and return its path.

    `image_rel_prefix` is relative to the PAGE, not to the run: pages written into
    `results/trajectories/` alongside `results/images/` want "../images". Nothing
    absolute is ever written into the HTML, so the folder can be moved or zipped.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = step_dicts(result)

    banner = ""
    if result.lied:
        banner = (
            '  <div class="false-success-banner">FALSE SUCCESS &mdash; this episode '
            'claimed a success it did not achieve.'
            '<small>The oracle measured failure after the scene settled. Nobody was '
            'told to look, which is what makes this the one failure a deployed robot '
            'cannot absorb.</small></div>')

    if steps is None:
        body_steps = (
            f'  <div class="notice">This results file carries <strong>no step '
            f'trace</strong> &mdash; only a step count of '
            f'<strong>{_esc(result.steps)}</strong>. '
            f'<code>EpisodeResult.steps</code> persists <code>len(trace.steps)</code>, '
            f'not the steps themselves, so a page rebuilt from '
            f'<code>episodes.jsonl</code> can show the outcome but not the frames. '
            f'Pass live <code>EpisodeResult</code> objects whose <code>steps</code> is '
            f'the list of step dicts to get the full trajectory.</div>')
        step_count = result.steps
    else:
        rows = "\n".join(_step_row(i, s, image_rel_prefix)
                         for i, s in enumerate(steps, start=1))
        body_steps = f'  <ol class="steps">\n{rows}\n  </ol>' if rows else (
            '  <div class="notice">This episode executed no steps.</div>')
        step_count = len(steps)

    body = "\n".join([
        '  <a class="back" href="../report.html">&larr; back to the report</a>',
        banner,
        f"  <h1>{_esc(result.episode_id)}</h1>",
        f'  <p class="muted">Instruction: &ldquo;{_esc(result.instruction)}&rdquo;</p>',
        _meta_grid(result, step_count),
        _outcome_chips(result),
        "  <h2>Trajectory</h2>",
        body_steps,
        '  <footer>Generated by <code>python -m harness.report</code>. Every number '
        'and every frame on this page comes from the episode record &mdash; nothing '
        'here is hand-written.</footer>',
    ])
    path = out_dir / f"{result.episode_id}.html"
    path.write_text(_page(result.episode_id, body), encoding="utf-8")
    return path
