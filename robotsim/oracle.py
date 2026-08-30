"""GROUND TRUTH. The judge's instrument, never the agent's.

Nothing under agent/ or primitives/ may import this module -- tests/test_firewall.py
enforces that with an AST scan. If you are tempted to import it from there to "help
the agent along", stop: that single shortcut invalidates every number in the report.
"""
from __future__ import annotations

import numpy as np

from robotsim.world import World


class Oracle:
    """Reads ground truth, and remembers the ORDER it saw things happen in.

    The order log is the one piece of state this class carries. It has to be here
    rather than derived at scoring time because placement order is not recoverable
    from a final snapshot: three cubes in a bowl look identical however they got
    there. The harness polls `observe_placements()` after every primitive, from the
    same hook and at the same points for both conditions, so the log is a property of
    the world's history and not of the policy that produced it.
    """

    def __init__(self, world: World):
        self._world = world
        self._scene = world.scene
        self._placement_order: list[str] = []

    def position_of(self, body: str) -> np.ndarray:
        return self._world.sim.get_base_position(body)

    def is_contained(self, item: str, container: str) -> bool:
        cx, cy, radius, height = self._world._bowl_centers[container]
        p = self.position_of(item)
        inside_xy = abs(p[0] - cx) <= radius and abs(p[1] - cy) <= radius
        resting_low = p[2] <= height + 0.02
        return bool(inside_xy and resting_low)

    # --- goal conditions -------------------------------------------------
    def pairs_satisfied(self) -> int:
        """How many required (item, container) placements currently hold."""
        return sum(1 for item, container in self._scene.success.pairs
                   if self.is_contained(item, container))

    def total_pairs(self) -> int:
        return len(self._scene.success.pairs)

    def _excluded_satisfied(self) -> int:
        """How many `excluded` pairs are correctly NOT holding."""
        return sum(1 for item, container in self._scene.success.excluded
                   if not self.is_contained(item, container))

    def progress(self) -> float:
        """Fraction of graded conditions satisfied, in [0, 1].

        This is the metric that makes the horizon experiment readable. Binary success
        on a four-block task hides the difference between "moved nothing" and "moved
        three of four and ran out of budget", which is exactly the difference a
        horizon ladder exists to show: multiplicative decay is visible in partial
        credit and invisible in a 0/1.

        On a `recall` scene the denominator includes the `excluded` conditions, so
        taking the pre-placed block back out earns its share of the credit. On every
        other scene `excluded` is empty and this is simply satisfied/total pairs.
        Ordering is deliberately NOT part of progress: it is reported separately as
        `order_correct`, because "right set, wrong order" and "wrong set" are
        different failures and averaging them together would hide both.
        """
        total = self.total_pairs() + len(self._scene.success.excluded)
        if not total:
            return 0.0
        return (self.pairs_satisfied() + self._excluded_satisfied()) / total

    # --- placement order -------------------------------------------------
    def observe_placements(self) -> list:
        """Poll containment and append anything newly placed. Idempotent per item.

        Called by the harness after every executed primitive. An item is logged the
        first time it is seen inside its target container and never again, so a block
        that is ejected by the disturbance and re-placed does not appear twice -- the
        log answers "in what order did these first arrive", which is what an ordering
        instruction actually asks.
        """
        for item, container in self._scene.success.pairs:
            if item not in self._placement_order and self.is_contained(item, container):
                self._placement_order.append(item)
        return list(self._placement_order)

    @property
    def placement_order(self) -> list:
        return list(self._placement_order)

    def order_correct(self) -> bool:
        """Is the order observed SO FAR consistent with the order required?

        Prefix comparison, not equality: an episode that placed red then green out of
        a required red-green-yellow has not yet broken the ordering, it has simply not
        finished. That distinction is what lets the report separate "got the set
        right, order wrong" from "did not finish", which is the whole reason mem_order
        exists. Full success additionally requires every pair, checked in
        `actual_success`.

        Vacuously True where the scene imposes no order -- only `ordered_pairs` scenes
        make this field meaningful.
        """
        if self._scene.success.type != "ordered_pairs":
            return True
        expected = [item for item, _ in self._scene.success.pairs]
        seen = self._placement_order
        return seen == expected[:len(seen)]

    # --- the verdict -----------------------------------------------------
    def actual_success(self, asked_human: bool) -> bool:
        """The single place a trajectory meets its goal.

        `asked_human` is still accepted because the harness records the escalation on
        every episode and passes it here uniformly; no current success type consults
        it. An unknown type raises rather than defaulting to False: a scene whose
        predicate this module does not implement must stop the run, not silently score
        zero for everyone and look like a hard task.
        """
        spec = self._scene.success
        total = self.total_pairs()
        all_pairs = bool(total and self.pairs_satisfied() == total)

        if spec.type == "pairs":
            return all_pairs
        if spec.type == "ordered_pairs":
            expected = [item for item, _ in spec.pairs]
            return bool(all_pairs and self._placement_order == expected)
        if spec.type == "recall":
            return bool(all_pairs
                        and self._excluded_satisfied() == len(spec.excluded))
        raise ValueError(f"unknown success type: {spec.type}")
