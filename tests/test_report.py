"""The report is the product, so the report is under test.

Every fixture here is synthetic and built in-process. Nothing reads results/ --
a test that depended on a real run would go red the first time the eval produced a
different number, which is the opposite of what a test is for, and it would also
make the report untestable before the first run ever finished.
"""
from __future__ import annotations

import json
import pathlib
import re
import xml.etree.ElementTree as ET

import pytest

from harness import report as report_mod
from harness import trajectory as trajectory_mod
from harness.metrics import EpisodeResult

PLACEHOLDERS = ("TBD", "TODO", "FIXME", "XXX", "lorem ipsum")


def make_step(primitive="grasp", *, image_path="results/images/ep_001_overhead.png",
              verdict=None, status="ok", error=None, fingers_width=0.044,
              reasoning="I will grasp the red cube.", note=None, detections=None):
    return {
        "primitive": primitive,
        "args": {"object_id": "red_cube_1"},
        "reasoning": reasoning,
        "feedback": {
            "primitive": primitive,
            "args": {"object_id": "red_cube_1"},
            "status": status,
            "error": error,
            "fingers_width": fingers_width,
            "ee_position": [0.02, -0.05, 0.16],
            "detections": detections if detections is not None else [
                {"id": "red_cube_1", "where": "centre"},
                {"id": "blue_bowl_1", "where": "top"},
            ],
            "image_path": image_path,
            "sim_steps": 140,
            "note": note,
        },
        "verdict": verdict,
    }


def make_result(condition="one_shot", scene_id="h1_single", seed=0,
                failure_mode="horizon_1", claimed=True, actual=True, *, steps=None,
                asked_human=False, recoveries=0, l3_calls=0, vlm_calls=3,
                cost_usd=0.005, drift=0, wall_seconds=6.5, progress=None,
                instruction="Put all the blocks in the blue bowl."):
    if steps is None:
        steps = [make_step("look", verdict=None)]
    return EpisodeResult(
        condition=condition, scene_id=scene_id, seed=seed, failure_mode=failure_mode,
        instruction=instruction, claimed_success=claimed, actual_success=actual,
        asked_human=asked_human, recoveries=recoveries, steps=steps,
        vlm_calls=vlm_calls, input_tokens=1000, output_tokens=100, cost_usd=cost_usd,
        drift=drift, episode_id=f"{condition}_{scene_id}_s{seed}",
        claim_reason="the block is in the bowl", stop_reason="agent called report_done",
        wall_seconds=wall_seconds, l3_calls=l3_calls,
        progress=(1.0 if actual else 0.0) if progress is None else progress,
        pairs_total=1,
    )


# Two conditions x every experimental cell, with a lie in the one-shot row of every
# cell and a clean agentic row to contrast it against. Small, but it exercises every
# code path the report has: deltas, per-scene grouping, L3 accounting, and the chart.
FAILURE_MODES = {
    "horizon_1": "h1_single",
    "horizon_2": "h2_pair",
    "horizon_3": "h3_triple",
    "matching_3": "match3",
    "memory_order": "mem_order",
    "memory_swap": "mem_swap",
    "memory_recall": "mem_recall",
    "disturbance": "disturb_h3",
}
# The cell the visual verifier is marked on, and the cell nothing can succeed at.
PROBE_MODE, PROBE_SCENE = "disturbance", report_mod.L3_PROBE_SCENE
HARD_MODE = "memory_swap"


def dataset() -> list[EpisodeResult]:
    results = []
    for condition in report_mod.CONDITION_ORDER:
        agentic = condition != "one_shot"
        for mode, scene in FAILURE_MODES.items():
            for seed in (0, 1):
                good = agentic and not (mode == HARD_MODE)
                claimed = True if not agentic else good
                actual = good and mode != HARD_MODE
                l3 = 2 if agentic else 0
                verdict = None
                if l3 and mode == PROBE_MODE:
                    verdict = {"ok": False, "layer": "L3",
                               "reason": "visual check failed: is the red cube in the "
                                         "blue bowl? -> no", "informational": False}
                # Distinct frames per episode: the evidence pack's promise is that it
                # copies only the images its own pages reference, and a fixture that
                # shared one file between every episode could not tell the difference.
                eid = f"{condition}_{scene}_s{seed}"
                frame = lambda i, e=eid: f"results/images/{e}_00{i}_overhead.png"
                results.append(make_result(
                    condition=condition, scene_id=scene, seed=seed,
                    failure_mode=mode, claimed=claimed, actual=actual,
                    l3_calls=l3, recoveries=1 if agentic else 0,
                    steps=[make_step("look", verdict=None, image_path=frame(1)),
                           make_step("grasp", verdict=verdict, image_path=frame(2))],
                ))
    return results


@pytest.fixture()
def generated(tmp_path):
    results = dataset()
    out = tmp_path / "out"
    paths = report_mod.generate(results, out)
    return results, out, paths


# --------------------------------------------------------------------------- tables

def test_headline_table_has_a_row_for_every_condition_present(generated):
    _, out, _ = generated
    markdown = (out / "report.md").read_text(encoding="utf-8")
    table = markdown.split("## ")[1]
    for condition in report_mod.CONDITION_ORDER:
        assert f"`{condition}`" in table, f"{condition} missing from headline table"


def test_headline_table_omits_conditions_that_did_not_run(tmp_path):
    results = [make_result(condition="one_shot")]
    report_mod.generate(results, tmp_path)
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "`agentic`" not in markdown


def test_honesty_gap_is_computed_and_rendered(tmp_path):
    # 5 episodes, all claimed, 2 actually succeeded -> 1.00 - 0.40 = +0.60.
    results = [make_result(condition="one_shot", seed=i, claimed=True, actual=i < 2)
               for i in range(5)]
    report_mod.generate(results, tmp_path)
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "0.60" in markdown
    assert "1.00 (5/5)" in markdown
    assert "0.40 (2/5)" in markdown


def test_comparison_reports_the_delta_the_loop_added(tmp_path):
    results = ([make_result(condition="one_shot", seed=i, claimed=True, actual=False)
                for i in range(4)]
               + [make_result(condition="agentic", seed=i, claimed=True, actual=i < 2)
                  for i in range(4)])
    report_mod.generate(results, tmp_path)
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    comparison = markdown.split("## ")[2]
    assert "what the loop bought" in comparison.lower()
    assert "+0.50" in comparison      # task success 0.00 -> 0.50
    assert "-0.50" in comparison      # honesty gap 1.00 -> 0.50


def test_per_failure_mode_breakdown_names_every_mode(generated):
    _, out, _ = generated
    markdown = (out / "report.md").read_text(encoding="utf-8")
    for mode in FAILURE_MODES:
        assert f"`{mode}`" in markdown
    for scene in FAILURE_MODES.values():
        assert scene in markdown


# ------------------------------------------------------------------------------ L3

def test_l3_section_appears_when_probe_episodes_are_present(generated):
    _, out, _ = generated
    markdown = (out / "report.md").read_text(encoding="utf-8")
    assert "Visual verifier error rate" in markdown
    assert PROBE_SCENE in markdown
    assert "false positive" in markdown.lower()
    assert "false negative" in markdown.lower()
    # The sampling caveat must survive into the prose; without it the denominator
    # reads as larger than it really is.
    assert "only fires after" in markdown


def test_l3_section_degrades_gracefully_without_probe_episodes(tmp_path):
    results = [make_result(condition="agentic", scene_id="h1_single",
                           failure_mode="horizon_1", seed=i, l3_calls=2)
               for i in range(3)]
    report_mod.generate(results, tmp_path)
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Visual verifier error rate" in markdown
    assert f"no `{PROBE_SCENE}` episodes" in markdown
    assert "nan" not in markdown.lower()


def test_l3_denominator_excludes_episodes_that_never_ran_l3(tmp_path):
    """A one-shot episode on the probe scene says nothing about the verifier."""
    results = [make_result(condition="one_shot", scene_id=PROBE_SCENE,
                           failure_mode=PROBE_MODE, seed=i, l3_calls=0)
               for i in range(4)]
    results += [make_result(condition="agentic", scene_id=PROBE_SCENE,
                            failure_mode=PROBE_MODE, seed=i, l3_calls=1)
                for i in range(2)]
    stats = report_mod.l3_error_stats(results)
    assert stats["episodes"] == 6
    assert stats["l3_episodes"] == 2
    assert stats["traced_episodes"] == 2
    assert stats["answers"] == 2


def test_l3_section_says_so_when_the_layer_was_never_switched_on(tmp_path):
    results = [make_result(condition="one_shot", scene_id=PROBE_SCENE,
                           failure_mode=PROBE_MODE, seed=i, l3_calls=0)
               for i in range(3)]
    report_mod.generate(results, tmp_path)
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "none of them switched L3 on" in markdown
    assert "false positives" not in markdown.lower()


def test_l3_section_declines_to_guess_without_step_traces(tmp_path):
    results = [make_result(condition="agentic", scene_id=PROBE_SCENE,
                           failure_mode=PROBE_MODE, seed=i, l3_calls=2, steps=6)
               for i in range(3)]
    report_mod.generate(results, tmp_path)
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "not computed" in markdown
    assert "nan" not in markdown.lower()


def test_l3_error_counts_come_from_the_verdicts(tmp_path):
    # One occluded episode that actually succeeded but whose verifier said "no":
    # exactly one false negative, and one "yes" answer (2 l3 calls, 1 recorded no)
    # on an episode that succeeded, which is a true positive, not a false one.
    no_verdict = {"ok": False, "layer": "L3", "reason": "visual check failed",
                  "informational": False}
    results = [make_result(condition="agentic", scene_id=PROBE_SCENE,
                           failure_mode=PROBE_MODE, claimed=True, actual=True,
                           l3_calls=2,
                           steps=[make_step("grasp", verdict=no_verdict),
                                  make_step("place", verdict=None)])]
    stats = report_mod.l3_error_stats(results)
    assert stats["answers"] == 2
    assert stats["no_answers"] == 1
    assert stats["yes_answers"] == 1
    assert stats["false_negatives"] == 1
    assert stats["false_positives"] == 0


# --------------------------------------------------------------------------- chart

def test_svg_chart_is_well_formed_and_has_one_group_per_condition(generated):
    results, out, _ = generated
    summaries = report_mod.summaries_for(results)
    svg = report_mod.honesty_chart_svg(summaries)
    root = ET.fromstring(svg)          # raises if the hand-emitted SVG is malformed
    assert root.tag.endswith("svg")
    groups = [g for g in root.iter() if g.get("class") == "bar-group"]
    assert len(groups) == len(summaries)
    assert "#000" not in svg and "black" not in svg   # must survive a dark ground
    assert "currentColor" in svg
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "<svg" in html


def test_chart_is_present_in_both_report_formats(generated):
    _, out, _ = generated
    assert "<svg" in (out / "report.md").read_text(encoding="utf-8")
    assert "<svg" in (out / "report.html").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- output

def test_both_report_formats_are_written(generated):
    _, out, paths = generated
    assert (out / "report.md").is_file()
    assert (out / "report.html").is_file()
    assert paths["markdown"] == out / "report.md"
    assert paths["html"] == out / "report.html"


def test_a_trajectory_page_is_written_for_every_episode(generated):
    results, out, paths = generated
    pages = sorted((out / "trajectories").glob("*.html"))
    assert len(pages) == len({r.episode_id for r in results})
    markdown = (out / "report.md").read_text(encoding="utf-8")
    assert "trajectories/" in markdown


def test_generated_markdown_has_no_placeholder_text(generated):
    _, out, _ = generated
    markdown = (out / "report.md").read_text(encoding="utf-8")
    for token in PLACEHOLDERS:
        assert token not in markdown, f"placeholder {token!r} leaked into report.md"


def test_inline_markdown_survives_into_html():
    out = report_mod._inline("a `code` and **bold** and *italic* and "
                             "[link](trajectories/x.html)")
    assert "<code>code</code>" in out
    assert "<strong>bold</strong>" in out
    assert "<em>italic</em>" in out
    assert '<a href="trajectories/x.html">link</a>' in out
    assert "*" not in out and "`" not in out
    # bold must not be shredded into an italic wrapping an asterisk
    assert "<em>" not in report_mod._inline("**only bold**")


def test_inline_markdown_cannot_inject_markup():
    out = report_mod._inline("<script>alert(1)</script> **x**")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_no_raw_asterisks_or_backticks_leak_into_the_html_report(generated):
    _, out, _ = generated
    body = (out / "report.html").read_text(encoding="utf-8").split("</style>")[1]
    assert "**" not in body
    assert "`" not in body


def test_html_entities_are_not_double_escaped(generated):
    """Text bound for a table cell is escaped on the way out, so an HTML entity
    written into it renders as the literal string `&mdash;` instead of a dash."""
    _, out, _ = generated
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "&amp;mdash;" not in html
    assert not re.search(r"&amp;[a-zA-Z]+;", html)


def test_generated_html_has_no_placeholder_text(generated):
    _, out, _ = generated
    html = (out / "report.html").read_text(encoding="utf-8")
    for token in PLACEHOLDERS:
        assert token not in html


def test_cli_reads_a_jsonl_file_and_writes_the_report(tmp_path):
    from harness.metrics import write_results
    source = tmp_path / "episodes.jsonl"
    # steps come back off disk as whatever was written; the int form is what the
    # current EpisodeResult schema persists, so the CLI must survive it.
    write_results(source, [make_result(steps=4), make_result(condition="agent",
                                                             steps=5)])
    out = tmp_path / "out"
    assert report_mod.main(["--results", str(source), "--out", str(out)]) == 0
    assert (out / "report.md").is_file()
    assert (out / "report.html").is_file()
    assert list((out / "trajectories").glob("*.html"))


# ---------------------------------------------------------------------- trajectory

def test_lying_episode_page_carries_the_false_success_banner(tmp_path):
    result = make_result(condition="baseline", claimed=True, actual=False)
    page = trajectory_mod.render_trajectory(result, tmp_path)
    html = page.read_text(encoding="utf-8")
    assert "FALSE SUCCESS" in html
    assert "false-success-banner" in html


def test_honest_episode_page_has_no_false_success_banner(tmp_path):
    result = make_result(claimed=False, actual=False)
    html = trajectory_mod.render_trajectory(result, tmp_path).read_text("utf-8")
    assert "FALSE SUCCESS" not in html
    assert "agent claimed" in html and "oracle measured" in html


def test_trajectory_renders_with_missing_image_and_missing_verdict(tmp_path):
    result = make_result(steps=[make_step("look", image_path=None, verdict=None),
                                make_step("grasp", image_path="", verdict=None)])
    html = trajectory_mod.render_trajectory(result, tmp_path).read_text("utf-8")
    assert "no frame" in html.lower()
    assert "not checked" in html.lower()
    assert "<img" not in html


def test_trajectory_image_sources_are_relative(tmp_path):
    result = make_result(steps=[make_step(
        "grasp", image_path="/absolute/results/images/ep_003_overhead.png")])
    html = trajectory_mod.render_trajectory(result, tmp_path).read_text("utf-8")
    assert 'src="images/ep_003_overhead.png"' in html
    assert "/absolute/" not in html


def test_trajectory_is_self_contained_no_network_assets(tmp_path):
    html = trajectory_mod.render_trajectory(make_result(), tmp_path).read_text("utf-8")
    for token in ("http://", "https://", "cdn", "<link"):
        assert token not in html.lower(), f"{token} would break the offline folder"


def test_trajectory_is_theme_aware(tmp_path):
    html = trajectory_mod.render_trajectory(make_result(), tmp_path).read_text("utf-8")
    assert "prefers-color-scheme: dark" in html
    assert ":root" in html


def test_verdict_badge_names_the_layer_that_objected(tmp_path):
    result = make_result(steps=[make_step("grasp", verdict={
        "ok": False, "layer": "L2", "reason": "gripper closed to 0.0004 m -- it "
        "grasped air", "informational": False})])
    html = trajectory_mod.render_trajectory(result, tmp_path).read_text("utf-8")
    assert "L2" in html
    assert "grasped air" in html


def test_informational_verdict_is_not_shown_as_a_failure(tmp_path):
    result = make_result(steps=[make_step("move_to", verdict={
        "ok": True, "layer": "L3", "reason": "(informational) is the cube in the "
        "bowl? -> no", "informational": True})])
    html = trajectory_mod.render_trajectory(result, tmp_path).read_text("utf-8")
    assert "informational" in html.lower()
    assert 'class="badge verdict-fail"' not in html


def test_aperture_is_classified_for_the_l2_signal(tmp_path):
    closed = trajectory_mod.aperture_class(0.0004)
    holding = trajectory_mod.aperture_class(0.044)
    empty = trajectory_mod.aperture_class(0.080)
    assert closed != holding != empty
    result = make_result(steps=[make_step("grasp", fingers_width=0.0004)])
    html = trajectory_mod.render_trajectory(result, tmp_path).read_text("utf-8")
    assert "closed on air" in html.lower()


def test_aperture_threshold_tracks_the_primitive_that_defines_it(tmp_path):
    """The report may not invent the L2 cliff; it must quote the one L2 uses."""
    from primitives.api import EMPTY_GRIP_THRESHOLD
    assert trajectory_mod.CLOSED_ON_AIR_M == EMPTY_GRIP_THRESHOLD


def test_trajectory_survives_a_results_file_without_step_traces(tmp_path):
    """`EpisodeResult.steps` is an int count in the persisted schema."""
    html = trajectory_mod.render_trajectory(make_result(steps=7),
                                            tmp_path).read_text("utf-8")
    assert "7" in html
    assert "no step trace" in html.lower()


def test_html_is_escaped_not_injected(tmp_path):
    result = make_result(steps=[make_step(
        "grasp", reasoning="<script>alert('x')</script> & then grasp")])
    html = trajectory_mod.render_trajectory(result, tmp_path).read_text("utf-8")
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html


# ----------------------------------------------------------------------- evidence

def test_evidence_pack_pairs_baseline_and_agent_for_each_failure_mode(tmp_path):
    results = dataset()
    source = tmp_path / "results"
    images = source / "images"
    images.mkdir(parents=True)
    for result in results:
        for step in result.steps:
            name = pathlib.Path(step["feedback"]["image_path"]).name
            (images / name).write_bytes(b"\x89PNG\r\n\x1a\n")
    report_mod.generate(results, source)
    evidence = tmp_path / "evidence"
    written = report_mod.write_evidence(results, source, evidence)
    index = (evidence / "index.html").read_text(encoding="utf-8")
    for mode in FAILURE_MODES:
        assert mode in index
    assert (evidence / "report.md").is_file()
    assert (evidence / "report.html").is_file()
    assert (evidence / "episodes.jsonl").is_file()
    # one baseline + one agent page per failure mode
    pages = sorted((evidence / "trajectories").glob("*.html"))
    assert len(pages) == 2 * len(FAILURE_MODES)
    assert written["episodes"] == len(pages)
    # only the images those pages reference travelled with them
    copied = list((evidence / "images").glob("*.png"))
    assert copied and len(copied) < len(list(images.glob("*.png")))
    # the jsonl is still readable by the reader that wrote it
    lines = (evidence / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["condition"]


def test_evidence_report_never_links_a_page_it_did_not_ship(tmp_path):
    """A dead link in the pack is worse than no link: it reads as a broken deliverable
    at exactly the moment someone is deciding whether to trust it."""
    results = dataset()
    source = tmp_path / "results"
    report_mod.generate(results, source)
    evidence = tmp_path / "evidence"
    report_mod.write_evidence(results, source, evidence)
    shipped = {p.name for p in (evidence / "trajectories").glob("*.html")}
    for text in ((evidence / "report.md").read_text(encoding="utf-8"),
                 (evidence / "report.html").read_text(encoding="utf-8")):
        linked = set(re.findall(r"trajectories/([A-Za-z0-9_]+\.html)", text))
        assert linked, "the pack's report links no trajectory pages at all"
        assert linked <= shipped, f"dead links: {sorted(linked - shipped)[:3]}"
    # the full report, by contrast, links every episode it rendered
    full = (source / "report.md").read_text(encoding="utf-8")
    assert len(set(re.findall(r"trajectories/([A-Za-z0-9_]+\.html)", full))) == \
        len(results)


def test_trajectory_prefers_the_persisted_trace_steps(tmp_path):
    """Regression: EpisodeResult.steps is an int count, so trajectory pages were
    rendering headers with no frames, no reasoning and no verdicts -- gutting a
    required deliverable. trace_steps carries the real trace."""
    from harness.metrics import EpisodeResult
    from harness.trajectory import render_trajectory, step_dicts

    step = {"primitive": "grasp", "args": {"object_id": "red_cube_1"},
            "reasoning": "the red cube is the target",
            "feedback": {"primitive": "grasp", "args": {}, "status": "ok", "error": None,
                         "fingers_width": 0.044, "ee_position": (0.0, 0.0, 0.2),
                         "detections": [], "image_path": None, "sim_steps": 3, "note": None},
            "verdict": {"ok": True, "layer": None, "reason": "", "informational": False}}
    result = EpisodeResult(
        condition="agentic", scene_id="h1_single", seed=0, failure_mode="horizon_1",
        instruction="Put the red block in the blue bowl.", claimed_success=True,
        actual_success=True, asked_human=False, recoveries=0, steps=1,
        vlm_calls=2, input_tokens=10, output_tokens=2, cost_usd=0.001, drift=0,
        episode_id="agentic_h1_single_s0", trace_steps=[step])

    assert step_dicts(result) == [step]
    page = render_trajectory(result, tmp_path)
    html = page.read_text()
    assert "the red cube is the target" in html
    assert "red_cube_1" in html
    assert "no step" not in html.lower()
