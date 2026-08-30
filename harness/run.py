"""The eval runner: every condition x every scene x every seed, one episode at a time.

Also the judge side of the replay contract. LLMClient.complete() reads the cache but
never writes it, and agent/ is frozen, so recording a live run is the harness's job --
see _RecordingClient. Without it a live run would spend money and leave nothing behind
for `make judge` to reproduce.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from agent.llm import CacheMiss, LLMClient
from agent.verify import VerificationConfig
from harness.episode import run_episode
from harness.metrics import write_results
from harness.scenes import load_scenes

SEEDS = [0, 1, 2, 3, 4]

# The KEY is the condition name: it is what the report rows say and what run_episode
# stamps on every EpisodeResult.
#
# KNOWN, DELIBERATE MISMATCH, documented rather than hidden: agent/react.py and
# agent/verify.py build VLMCall.condition from `config.label`, not from this key. For
# four of the five rows the two agree; for "agent" they do not -- its label is
# "agent_L1L2L3", so its cache files are named ..._agent_L1L2L3_s0_001_plan.json while
# the report row reads "agent". That is a legibility wart, not a correctness bug: the
# label is a pure function of the config, so the same key is produced on the recording
# run and the replay run, and all five labels are distinct ("baseline", "agent_L1",
# "agent_L1L2", "agent_L1L2L3", "agent_verify_every") so no two conditions can collide
# in the cache. Renaming the key to "agent_L1L2L3" would fix the wart, and closing it
# from the other side would mean editing agent/, which is out of scope here.
CONDITIONS = {
    "baseline":           None,
    "agent_L1":           VerificationConfig(l1=True,  l2=False, l3=False),
    "agent_L1L2":         VerificationConfig(l1=True,  l2=True,  l3=False),
    "agent":              VerificationConfig(l1=True,  l2=True,  l3=True),
    "agent_verify_every": VerificationConfig(l1=True, l2=True, l3=True, verify_every_primitive=True),
}


class _RecordingClient:
    """An LLMClient that also persists what it paid for.

    Every live response is written to the replay cache as it arrives, not at the end
    of the run: an episode that crashes half way has still banked the calls it made,
    so a re-run resumes from the cache instead of buying them again.
    """

    def __init__(self, client: LLMClient):
        self._client = client

    def complete(self, call):
        response = self._client.complete(call)
        if self._client.mode == "live":
            self._client.write_cache(call, response)
        return response

    def __getattr__(self, name):
        return getattr(self._client, name)


def _parse_list(value: str, allowed, what: str) -> list:
    if value == "all":
        return list(allowed)
    picked = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in picked if item not in allowed]
    if unknown:
        raise SystemExit(f"unknown {what}: {', '.join(unknown)}. "
                         f"known: {', '.join(map(str, allowed))}")
    return picked


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the eval and score every episode.")
    parser.add_argument("--conditions", default="all",
                        help=f"comma-separated, or 'all'. {', '.join(CONDITIONS)}")
    parser.add_argument("--mode", default="replay", choices=("replay", "live"),
                        help="replay reads the committed cache; live calls the VLM")
    parser.add_argument("--scenes", default="all", help="comma-separated scene ids, or 'all'")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)),
                        help="comma-separated seeds")
    parser.add_argument("--out", default="results", help="output directory")
    args = parser.parse_args(argv)

    scenes = {s.id: s for s in load_scenes()}
    conditions = _parse_list(args.conditions, CONDITIONS, "condition")
    scene_ids = _parse_list(args.scenes, scenes, "scene")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_dir = pathlib.Path(args.out)

    results, misses = [], 0
    for condition in conditions:
        config = CONDITIONS[condition]
        for scene_id in scene_ids:
            for seed in seeds:
                # A fresh client per episode, so tokens and cost are per-episode
                # numbers rather than a running total the report would have to diff.
                client = _RecordingClient(LLMClient(args.mode, cache_dir="cache"))
                try:
                    result = run_episode(scenes[scene_id], seed, client, config,
                                         out_dir, condition=condition)
                except CacheMiss as exc:
                    # One missing entry kills one episode, never the run. The count
                    # comes back as a non-zero exit code so a partial replay can
                    # never be mistaken for a clean one.
                    misses += 1
                    print(f"MISS {condition} {scene_id} s{seed}: {exc}",
                          file=sys.stderr)
                    continue
                results.append(result)
                flag = "LIE" if result.lied else ("ok " if result.actual_success else "-- ")
                print(f"{flag} {result.condition:<18} {result.scene_id:<22} s{result.seed} "
                      f"claimed={str(result.claimed_success):<5} "
                      f"actual={str(result.actual_success):<5} "
                      f"steps={result.steps:<3} ${result.cost_usd:.4f}", flush=True)

    path = write_results(out_dir / "episodes.jsonl", results)
    lies = sum(1 for r in results if r.lied)
    print(f"\n{len(results)} episodes -> {path}  |  {lies} false success(es)  |  "
          f"${sum(r.cost_usd for r in results):.4f}  |  "
          f"drift {sum(r.drift for r in results)}  |  {misses} cache miss(es)")
    return 1 if misses else 0


if __name__ == "__main__":
    raise SystemExit(main())
