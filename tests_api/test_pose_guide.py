"""The sex-position guide: every entry complete, names matched the way briefs write them."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawk_api import pose_guide  # noqa: E402

FIELDS = ("tags:", "her:", "him:", "hands:", "faces:", "camera:", "motion:", "watch:", "prompt:")


class PoseGuide(unittest.TestCase):
    def test_every_position_carries_every_field(self):
        rules, poses = pose_guide.load()
        self.assertIn("know no position names", rules)
        self.assertIn("the man is out of frame", rules)
        self.assertGreaterEqual(len(poses), 20)
        for pose in poses:
            for field in FIELDS:
                self.assertIn(field, pose.text, f"{pose.key} lacks {field}")
        self.assertEqual(len({p.key for p in poses}), len(poses), "keys are unique")

    def test_the_longest_name_wins(self):
        keys = lambda text: [p.key for p in pose_guide.mentioned(text)]
        self.assertEqual(keys("Standing Full Nelson Anal"), ["full-nelson-standing"])
        self.assertEqual(keys("lying full nelson"), ["full-nelson-lying"])
        self.assertEqual(keys("reverse cowgirl, then doggy"), ["doggy", "cowgirl"])
        self.assertEqual(keys("a walk on the beach"), [])

    def test_lookup_lists_matches_or_the_index(self):
        self.assertEqual({p["key"] for p in pose_guide.lookup("nelson")["positions"]},
                         {"full-nelson-lying", "full-nelson-standing"})
        self.assertIn("note", pose_guide.lookup(None))
        self.assertIn("error", pose_guide.lookup("a sunset walk"))

    def test_the_planner_gets_the_guide_only_for_a_sexual_brief(self):
        self.assertEqual(pose_guide.planner_note("A walk in the park at golden hour"), "")
        named = pose_guide.planner_note("Wall lift in the shower")
        self.assertIn("Standing carry / Wall lift", named)
        self.assertNotIn("Butterfly", named)
        unnamed = pose_guide.planner_note("They have sex on the sofa")
        self.assertIn("Butterfly", unnamed, "no position named: the planner sees them all to choose from")
        self.assertLess(len(unnamed), 18000, "but only each one's sentence and pitfall")


if __name__ == "__main__":
    unittest.main()
