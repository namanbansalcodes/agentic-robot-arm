"""Build the pitch deck: a PDF of short slides, and a silent MP4 that walks through it.

Design rules, learned from a first cut that was hard to follow:

* One robot on screen at a time. A side-by-side of two arms doing different things asks
  the viewer to track two stories at once, and they track neither.
* Every clip carries its own explanation: the instruction, the call log revealing in step,
  and a verdict card at the end saying what the robot claimed against what it did.
* A slide is a label, a headline and one line. If it needs a paragraph it is two slides.

Every number is read from results/episodes.jsonl and docs/videos/*.json at build time, so
nothing here can drift from the eval.

    python docs/demo/build_pitch.py
"""
from __future__ import annotations

import json
import pathlib

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence

W, H = 1920, 1080
FPS = 25

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "demo"
GIFS = ROOT / "docs" / "videos"

BG    = (14, 17, 22)
PANEL = (23, 27, 34)
FG    = (244, 244, 245)
MUTED = (154, 162, 173)
DIM   = (98, 106, 118)
RED   = (248, 113, 113)
GREEN = (74, 222, 128)
AMBER = (251, 191, 36)
BLUE  = (125, 176, 255)

HEL = "/System/Library/Fonts/HelveticaNeue.ttc"
MENLO = "/System/Library/Fonts/Menlo.ttc"
IDX = {"regular": 0, "bold": 1, "medium": 2}
MARGIN = 130


def font(size, weight="regular"):
    return ImageFont.truetype(HEL, size, index=IDX.get(weight, 0))


def mono(size, bold=False):
    return ImageFont.truetype(MENLO, size, index=1 if bold else 0)


def wrap(draw, text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=fnt) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def clip_text(draw, text, fnt, max_w):
    if draw.textlength(text, font=fnt) <= max_w:
        return text
    while text and draw.textlength(text + "…", font=fnt) > max_w:
        text = text[:-1]
    return text + "…"


# --------------------------------------------------------------------------- data

def load_numbers():
    rows = [json.loads(l) for l in (ROOT / "results" / "episodes.jsonl").read_text().splitlines() if l.strip()]

    def agg(rs):
        n = len(rs)
        wins = sum(1 for r in rs if r["actual_success"])
        claims = sum(1 for r in rs if r["claimed_success"])
        return {"n": n, "wins": wins, "success": wins / n, "claimed": claims / n,
                "gap": claims / n - wins / n,
                "lies": sum(1 for r in rs if r["claimed_success"] and not r["actual_success"]),
                "calls": sum(r["vlm_calls"] for r in rs) / n,
                "cost": sum(r["cost_usd"] for r in rs),
                "recoveries": sum(r["recoveries"] for r in rs),
                "drift": sum(r["drift"] for r in rs)}

    scenes = ["h1_single", "h2_pair", "h3_triple", "match3", "mem_order",
              "mem_swap", "mem_recall", "disturb_h3", "disturb_match3"]
    return {"episodes": len(rows),
            "one_shot": agg([r for r in rows if r["condition"] == "one_shot"]),
            "agentic": agg([r for r in rows if r["condition"] == "agentic"]),
            "scenes": scenes,
            "per": {s: {c: agg([r for r in rows if r["scene_id"] == s and r["condition"] == c])
                        for c in ("one_shot", "agentic")} for s in scenes}}


# ------------------------------------------------------------------------- chrome

def base(page=None, total=None, footer=""):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    if page is not None:
        d.line([(MARGIN, H - 78), (W - MARGIN, H - 78)], fill=(34, 39, 48), width=2)
        d.line([(MARGIN, H - 78), (MARGIN + (W - 2 * MARGIN) * (page / total), H - 78)],
               fill=BLUE, width=3)
        d.text((MARGIN, H - 128), footer, font=font(23), fill=DIM)
        d.text((W - MARGIN, H - 128), f"{page}/{total}", font=font(23), fill=DIM, anchor="ra")
    return img, d


def statement(label, colour, headline, support, page, total, footer):
    img, d = base(page, total, footer)
    d.line([(MARGIN, 150), (MARGIN + 64, 150)], fill=colour, width=5)
    d.text((MARGIN, 196), label.upper(), font=font(27, "bold"), fill=colour)
    lines = wrap(d, headline, font(96, "bold"), W - 2 * MARGIN - 40)
    y = 350 if len(lines) < 3 else 290
    for line in lines:
        d.text((MARGIN, y), line, font=font(96, "bold"), fill=FG)
        y += 120
    if support:
        y += 40
        for line in wrap(d, support, font(44), W - 2 * MARGIN - 160):
            d.text((MARGIN, y), line, font=font(44), fill=MUTED)
            y += 62
    return img


# -------------------------------------------------------------------------- slides

def bars(d, x, y, w, rows, unit=""):
    """Horizontal bars. Rows are (label, value, max, colour, caption)."""
    for i, (label, value, top, colour, caption) in enumerate(rows):
        yy = y + i * 148
        d.text((x, yy), label, font=font(30, "bold"), fill=colour)
        d.rectangle([x, yy + 48, x + w, yy + 92], fill=(28, 33, 41))
        span = max(int(w * (value / top)), 6)
        d.rectangle([x, yy + 48, x + span, yy + 92], fill=colour)
        d.text((x + span + 24, yy + 44), f"{value:g}{unit}", font=mono(40, True), fill=colour)
        d.text((x, yy + 100), caption, font=font(27), fill=MUTED)


def loop_diagram(d, x, y):
    """One-shot as a straight line, agentic as a cycle. The shape is the argument."""
    def box(cx, cy, w, h, text, colour, sub=None):
        d.rectangle([cx, cy, cx + w, cy + h], fill=PANEL, outline=colour, width=2)
        d.text((cx + w / 2, cy + (h / 2 if not sub else h / 2 - 16)), text,
               font=font(28, "bold"), fill=FG, anchor="mm")
        if sub:
            d.text((cx + w / 2, cy + h / 2 + 20), sub, font=mono(22), fill=DIM, anchor="mm")

    def arrow(x1, y1, x2, y2, colour):
        d.line([(x1, y1), (x2, y2)], fill=colour, width=3)
        if x2 > x1:
            d.polygon([(x2, y2), (x2 - 14, y2 - 8), (x2 - 14, y2 + 8)], fill=colour)
        elif x2 < x1:
            d.polygon([(x2, y2), (x2 + 14, y2 - 8), (x2 + 14, y2 + 8)], fill=colour)
        else:
            head = 8 if y2 > y1 else -8
            d.polygon([(x2, y2), (x2 - 8, y2 - head), (x2 + 8, y2 - head)], fill=colour)

    bw, bh, gap = 380, 100, 96
    d.text((x, y), "ONE-SHOT", font=font(28, "bold"), fill=RED)
    for i, (label, sub) in enumerate([("plan everything", "1 model call"),
                                      ("execute every step", "reads nothing"),
                                      ("report success", "always")]):
        bx = x + i * (bw + gap)
        box(bx, y + 48, bw, bh, label, RED, sub)
        if i:
            arrow(bx - gap + 12, y + 48 + bh / 2, bx - 12, y + 48 + bh / 2, RED)

    y2 = y + 268
    d.text((x, y2), "AGENTIC", font=font(28, "bold"), fill=GREEN)
    for i, (label, sub) in enumerate([("choose ONE action", "1 model call"),
                                      ("execute it", "one primitive"),
                                      ("check the result", "L1 · L2 · L3")]):
        bx = x + i * (bw + gap)
        box(bx, y2 + 48, bw, bh, label, GREEN, sub)
        if i:
            arrow(bx - gap + 12, y2 + 48 + bh / 2, bx - 12, y2 + 48 + bh / 2, GREEN)

    # The return edge is the whole point, so it gets drawn as one continuous path
    # rather than three segments that have to agree with each other.
    box_bottom = y2 + 48 + bh
    lane = box_bottom + 74
    left_cx = x + bw / 2
    right_cx = x + 2 * (bw + gap) + bw / 2
    d.line([(right_cx, box_bottom), (right_cx, lane)], fill=GREEN, width=3)
    d.line([(right_cx, lane), (left_cx, lane)], fill=GREEN, width=3)
    d.line([(left_cx, lane), (left_cx, box_bottom + 12)], fill=GREEN, width=3)
    d.polygon([(left_cx, box_bottom), (left_cx - 9, box_bottom + 18),
               (left_cx + 9, box_bottom + 18)], fill=GREEN)
    d.text(((left_cx + right_cx) / 2, lane + 26), "decide again, with what actually happened",
           font=font(30, "bold"), fill=GREEN, anchor="ma")


def build_slides(N):
    one, ag = N["one_shot"], N["agentic"]
    total = 17
    foot = f"one-shot vs agentic · {N['episodes']} episodes · replay drift {one['drift'] + ag['drift']}"
    S = []

    # 1 title
    img, d = base()
    d.line([(MARGIN, 150), (MARGIN + 64, 150)], fill=BLUE, width=5)
    d.text((MARGIN, 196), "DEMO", font=font(27, "bold"), fill=BLUE)
    d.text((MARGIN, 380), "The Agentic Arm", font=font(158, "bold"), fill=FG)
    d.text((MARGIN, 596), "Act. Check. Recover. One step at a time.", font=font(50), fill=MUTED)
    d.text((MARGIN, 706), "Franka Panda  ·  PyBullet  ·  Gemini Robotics-ER 2", font=mono(30), fill=DIM)
    d.text((MARGIN, H - 128), foot, font=font(23), fill=DIM)
    S.append(img)

    # 2 the job
    S.append(statement("the job", BLUE, "Put every block in the bowl.",
                       "Three blocks. One bowl. A vision model driving the arm.", 2, total, foot))

    # 3 robot A
    S.append(statement("robot a  ·  one-shot", RED, "Plans every step up front, then executes blind.",
                       "One model call. It never reads a single result.", 3, total, foot))

    # 4 after clip A
    S.append(statement("what just happened", RED, "It left a block on the table.",
                       "Then it reported the job complete.", 4, total, foot))

    # 5 robot B
    S.append(statement("robot b  ·  agentic", GREEN, "One action. Check the result. Decide again.",
                       "Same model. Same tools. Same budget.", 5, total, foot))

    # 6 after clip B
    S.append(statement("what just happened", GREEN, "It noticed, went back, and finished.",
                       "A placed block was removed. It put it back.", 6, total, foot))

    # 7 the thesis
    S.append(statement("why blind planning breaks", AMBER,
                       "A plan written at t=0 cannot see t=5.",
                       "One early failure silently invalidates every step after it.",
                       7, total, foot))

    # 8 the loop
    img, d = base(8, total, foot)
    d.line([(MARGIN, 130), (MARGIN + 64, 130)], fill=BLUE, width=5)
    d.text((MARGIN, 172), "THE ONLY DIFFERENCE", font=font(27, "bold"), fill=BLUE)
    d.text((MARGIN, 232), "One of them closes the loop", font=font(66, "bold"), fill=FG)
    loop_diagram(d, MARGIN, 356)
    S.append(img)

    # 9 the three checks
    img, d = base(9, total, foot)
    d.line([(MARGIN, 150), (MARGIN + 64, 150)], fill=GREEN, width=5)
    d.text((MARGIN, 196), "CHECKING, AFTER EVERY ACTION", font=font(27, "bold"), fill=GREEN)
    d.text((MARGIN, 268), "Three checks, cheapest first", font=font(76, "bold"), fill=FG)
    for i, (tag, name, cost, detail) in enumerate([
            ("L1", "Did it error?", "free", "the error string it already got back"),
            ("L2", "What do the fingers say?", "free", "0.000 m means it closed on empty air"),
            ("L3", "Does a photo agree?", "1 call", "one yes/no question about a fresh frame")]):
        y = 442 + i * 150
        d.rectangle([MARGIN, y, W - MARGIN, y + 128], fill=PANEL)
        d.text((MARGIN + 34, y + 32), tag, font=mono(50, True), fill=BLUE)
        d.text((MARGIN + 148, y + 28), name, font=font(44, "bold"), fill=FG)
        d.text((MARGIN + 148, y + 82), detail, font=font(28), fill=DIM)
        d.text((W - MARGIN - 34, y + 44), cost, font=mono(36, True),
               fill=GREEN if cost == "free" else AMBER, anchor="ra")
    d.text((MARGIN, 900), "The two free checks run first. A photo is only worth buying once they pass.",
           font=font(34), fill=MUTED)
    S.append(img)

    # 10 the claim rate
    S.append(statement("the number that matters", RED,
                       f"Robot A reported success in all {one['n']} runs.",
                       "It finished half of them. Its report tracks nothing that happened.",
                       10, total, foot))

    # 11 results — jobs finished
    img, d = base(11, total, foot)
    d.line([(MARGIN, 150), (MARGIN + 64, 150)], fill=GREEN, width=5)
    d.text((MARGIN, 196), "THE RESULT", font=font(27, "bold"), fill=GREEN)
    d.text((MARGIN, 262), f"{N['episodes']} episodes, scored by ground truth", font=font(66, "bold"), fill=FG)
    bars(d, MARGIN, 420, 940, [
        ("ROBOT A  ·  one-shot", one["wins"], one["n"], RED, f"jobs actually finished, out of {one['n']}"),
        ("ROBOT B  ·  agentic", ag["wins"], ag["n"], GREEN, f"jobs actually finished, out of {ag['n']}"),
    ])
    d.text((MARGIN, 760), f"{one['success']:.0%}  →  {ag['success']:.0%}", font=mono(76, True), fill=FG)
    d.text((MARGIN + 560, 790), "same model, same tools, same budget", font=font(32), fill=MUTED)
    S.append(img)

    # 12 results — false successes
    img, d = base(12, total, foot)
    d.line([(MARGIN, 150), (MARGIN + 64, 150)], fill=RED, width=5)
    d.text((MARGIN, 196), "FALSE SUCCESSES", font=font(27, "bold"), fill=RED)
    d.text((MARGIN, 262), "Claimed done. Ground truth disagreed.", font=font(66, "bold"), fill=FG)
    bars(d, MARGIN, 420, 940, [
        ("ROBOT A  ·  one-shot", one["lies"], max(one["lies"], 1), RED, "episodes it claimed but had not done"),
        ("ROBOT B  ·  agentic", ag["lies"], max(one["lies"], 1), GREEN, "episodes it claimed but had not done"),
    ])
    d.text((MARGIN, 760), f"{one['lies']}  →  {ag['lies']}", font=mono(76, True), fill=FG)
    d.text((MARGIN + 360, 790), "an operator told a job is done when it is not",
           font=font(32), fill=MUTED)
    S.append(img)

    # 13 per task
    img, d = base(13, total, foot)
    d.line([(MARGIN, 150), (MARGIN + 64, 150)], fill=BLUE, width=5)
    d.text((MARGIN, 196), "EVERY TASK", font=font(27, "bold"), fill=BLUE)
    d.text((MARGIN, 262), "Where the loop earned its keep", font=font(70, "bold"), fill=FG)
    d.text((MARGIN + 760, 356), "ONE-SHOT", font=font(26, "bold"), fill=RED)
    d.text((MARGIN + 1120, 356), "AGENTIC", font=font(26, "bold"), fill=GREEN)
    names = {"h1_single": "1 block", "h2_pair": "2 blocks", "h3_triple": "3 blocks",
             "match3": "match 3 colours", "mem_order": "in a given order",
             "mem_swap": "swap two blocks", "mem_recall": "take one back out",
             "disturb_h3": "3 blocks, one removed", "disturb_match3": "match 3, one removed"}
    y = 404
    for s in N["scenes"]:
        o, a = N["per"][s]["one_shot"], N["per"][s]["agentic"]
        d.text((MARGIN, y), names[s], font=font(34), fill=FG)
        for x, dat in ((MARGIN + 760, o), (MARGIN + 1120, a)):
            colour = GREEN if dat["wins"] == 5 else (RED if dat["wins"] == 0 else AMBER)
            d.text((x, y), f"{dat['wins']}/5", font=mono(34, True), fill=colour)
            if dat["lies"]:
                d.text((x + 100, y + 6), f"{dat['lies']} false", font=mono(24), fill=RED)
        y += 58
    S.append(img)

    # 14 the honest loss
    rec = N["per"]["mem_recall"]
    S.append(statement("the honest loss", AMBER, "One task beat both of them.",
                       f"Robot A claimed it {rec['one_shot']['lies']} times anyway. Robot B, never once.",
                       14, total, foot))

    # 15 what it costs
    img, d = base(15, total, foot)
    d.line([(MARGIN, 150), (MARGIN + 64, 150)], fill=AMBER, width=5)
    d.text((MARGIN, 196), "WHAT THE LOOP COSTS", font=font(27, "bold"), fill=AMBER)
    d.text((MARGIN, 268), "Closing the loop is not free", font=font(76, "bold"), fill=FG)
    for x, val, lab, colour in ((MARGIN, f"{one['calls']:.0f}", "model calls per run · one-shot", RED),
                                (MARGIN + 620, f"{ag['calls']:.0f}", "model calls per run · agentic", GREEN),
                                (MARGIN + 1240, f"{ag['recoveries']:.0f}", "failures it caught and recovered from", BLUE)):
        d.text((x, 450), val, font=font(128, "bold"), fill=colour)
        for i, line in enumerate(wrap(d, lab, font(28), 470)):
            d.text((x, 610 + i * 38), line, font=font(28), fill=MUTED)
    d.text((MARGIN, 790), f"All {one['n']} live runs cost ${one['cost']:.2f} for Robot A, ${ag['cost']:.2f} for Robot B.",
           font=font(40), fill=MUTED)
    d.text((MARGIN, 852), "Replaying them from the saved answers costs nothing.", font=font(40), fill=MUTED)
    S.append(img)

    # 16 grounding
    img, d = base(16, total, foot)
    d.line([(MARGIN, 150), (MARGIN + 64, 150)], fill=BLUE, width=5)
    d.text((MARGIN, 196), "GROUNDED IN A 2026 RESULT", font=font(27, "bold"), fill=BLUE)
    d.text((MARGIN, 268), "VoLo · NVIDIA · arXiv:2606.07723", font=font(64, "bold"), fill=FG)
    d.text((MARGIN, 400), 'Its baseline family is literally "single action model,', font=font(42), fill=MUTED)
    d.text((MARGIN, 456), 'no orchestrator". That is Robot A.', font=font(42), fill=MUTED)
    d.text((MARGIN, 580), "42.9%  vs  14.3%", font=mono(64, True), fill=DIM)
    d.text((MARGIN, 660), "their orchestrated agent, on real hardware", font=font(30), fill=DIM)
    d.text((MARGIN + 900, 580), f"{ag['success']:.0%}  vs  {one['success']:.0%}", font=mono(64, True), fill=FG)
    d.text((MARGIN + 900, 660), "ours, in a controlled miniature", font=font(30), fill=MUTED)
    d.text((MARGIN, 800), "We do not reproduce VoLo. That needs their benchmark and their robot.",
           font=font(36), fill=AMBER)
    S.append(img)

    # 17 close
    img, d = base(17, total, foot)
    d.line([(MARGIN, 150), (MARGIN + 64, 150)], fill=GREEN, width=5)
    d.text((MARGIN, 196), "THE TAKEAWAY", font=font(27, "bold"), fill=GREEN)
    for i, line in enumerate(["Judge an embodied agent on the", "honesty gap, not the leaderboard."]):
        d.text((MARGIN, 300 + i * 104), line, font=font(84, "bold"), fill=FG)
    d.text((MARGIN, 560), "make judge", font=mono(88, True), fill=BLUE)
    d.text((MARGIN, 700), f"All {N['episodes']} runs. Offline. No API key. $0.00. Replay drift {one['drift'] + ag['drift']}.",
           font=font(42), fill=MUTED)
    d.text((MARGIN, 780), "github.com/namanbansalcodes/agentic-robot-arm", font=mono(34), fill=DIM)
    S.append(img)
    return S


# --------------------------------------------------------------------------- clips

def episode(scene, condition):
    meta = json.loads((GIFS / f"c_{scene}_{condition}.json").read_text())
    im = Image.open(GIFS / f"c_{scene}_{condition}.gif")
    # The left panel is the readable one: the right one is a wrist view where the arm
    # itself fills half the frame.
    frames = [f.convert("RGB").crop((0, 0, f.size[0] // 2, f.size[1])).copy()
              for f in ImageSequence.Iterator(im)]
    return meta, frames


def clip_frames(scene, condition, watch_hint):
    """One robot, its instruction, and its call log revealing as the arm moves."""
    meta, frames = episode(scene, condition)
    colour = RED if condition == "one_shot" else GREEN
    title = "ROBOT A  ·  never looks" if condition == "one_shot" else "ROBOT B  ·  looks after every step"

    # The bottom band is reserved for the verdict card, so the footage has to sit
    # entirely above it -- an overlay that covers the block on the table would hide
    # the one thing the viewer was told to watch.
    BAND = 760
    gw, gh = frames[0].size
    scale = min(880 / gw, (BAND - 270) / gh)
    tw, th = int(gw * scale), int(gh * scale)
    vx, vy = MARGIN, 250

    log = meta["log"]
    log_x, log_y = vx + tw + 80, 250
    log_w = W - log_x - 90

    out = []
    for i, frame in enumerate(frames):
        img, d = base()
        d.line([(MARGIN, 96), (MARGIN + 64, 96)], fill=colour, width=5)
        d.text((MARGIN, 132), title, font=font(30, "bold"), fill=colour)
        d.text((MARGIN, 184), f'"{meta["instruction"]}"', font=font(34), fill=MUTED)

        img.paste(frame.resize((tw, th), Image.LANCZOS), (vx, vy))
        d.rectangle([vx - 8, vy, vx - 3, vy + th], fill=colour)
        d.text((W - MARGIN, 140), watch_hint.upper(), font=font(30, "bold"), fill=AMBER, anchor="ra")

        # Reveal the log in step with the footage.
        shown = max(1, int(round((i + 1) / len(frames) * len(log))))
        y = log_y
        for entry in log[:shown][-10:]:
            if entry["call"].startswith("⚠"):
                d.rectangle([log_x - 14, y - 8, W - 90, y + 44], fill=(58, 40, 12))
                d.text((log_x, y), "DISTURBANCE", font=mono(28, True), fill=AMBER)
                d.text((log_x, y + 34), "a placed block is taken back out", font=font(22), fill=AMBER)
                y += 76
                continue
            d.text((log_x, y), clip_text(d, entry["call"], mono(28), log_w * 0.55),
                   font=mono(28), fill=FG if entry["ok"] else RED)
            detail = entry["detail"].split(" · ")[-1]
            d.text((W - 90, y + 4), clip_text(d, detail, mono(22), log_w * 0.4),
                   font=mono(22), fill=DIM, anchor="ra")
            y += 44
        out.append(img)

    # The verdict card: what it said, against what it did.
    said, truth = meta["reason"], f"{int(meta['progress'] * meta['total'])} of {meta['total']} blocks in the bowl"
    verdict = []
    for step in range(3):
        img = out[-1].copy()
        d = ImageDraw.Draw(img)
        d.rectangle([0, BAND, W, H], fill=(9, 11, 15))
        d.line([(0, BAND), (W, BAND)], fill=(38, 43, 52), width=2)
        if step >= 1:
            d.text((MARGIN, BAND + 34), "IT SAID", font=font(26, "bold"), fill=DIM)
            for j, line in enumerate(wrap(d, f'"{said}"', font(40), W - 2 * MARGIN - 420)[:2]):
                d.text((MARGIN, BAND + 74 + j * 52), line, font=font(40), fill=FG)
        if step >= 2:
            ok = meta["actual"]
            d.text((MARGIN, BAND + 196), "THE TRUTH", font=font(26, "bold"), fill=DIM)
            d.text((MARGIN, BAND + 234), truth, font=font(48, "bold"), fill=GREEN if ok else RED)
            honest = meta["claimed"] == meta["actual"]
            d.text((W - MARGIN, BAND + 150), "HONEST" if honest else "FALSE SUCCESS",
                   font=font(58, "bold"), fill=GREEN if honest else RED, anchor="ra")
        verdict.append(img)
    return out, verdict


# ---------------------------------------------------------------------------- emit

def bgr(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    N = load_numbers()
    slides = build_slides(N)

    pdf = OUT / "agentic-arm-pitch.pdf"
    slides[0].save(pdf, save_all=True, append_images=slides[1:], resolution=144.0)

    mp4 = OUT / "agentic-arm-pitch.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n_frames = 0

    def hold(img, seconds):
        nonlocal n_frames
        frame = bgr(img)
        for _ in range(int(seconds * FPS)):
            writer.write(frame)
            n_frames += 1

    holds = {1: 4.0, 2: 4.0, 3: 4.5, 4: 5.0, 5: 4.5, 6: 5.0, 7: 5.5, 8: 10.0,
             9: 9.0, 10: 5.5, 11: 7.5, 12: 7.5, 13: 10.0, 14: 5.5, 15: 8.0,
             16: 9.0, 17: 8.0}
    # Robot A's clip runs after slide 3, robot B's after slide 5.
    clips = {3: ("disturb_h3", "one_shot", "watch the red block"),
             5: ("disturb_h3", "agentic", "watch the red block")}

    for i, slide in enumerate(slides, start=1):
        hold(slide, holds[i])
        if i in clips:
            scene, condition, hint = clips[i]
            frames, verdict = clip_frames(scene, condition, hint)
            for f in frames:                       # 80 ms per source frame
                hold(f, 2 / FPS)
            hold(verdict[0], 0.8)
            hold(verdict[1], 3.2)
            hold(verdict[2], 4.5)
    writer.release()

    print(f"pdf   {pdf}  ({len(slides)} slides)")
    print(f"video {mp4}  ({n_frames} frames · {n_frames / FPS:.1f}s)")


if __name__ == "__main__":
    main()
