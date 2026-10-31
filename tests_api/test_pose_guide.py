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
        self.assertEqual(keys("reverse cowgirl, then doggy"), ["doggy", "reverse-cowgirl"])
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
        self.assertIn("Doggy style", unnamed, "no position named: the core ones in full to choose from")
        self.assertIn("Butterfly", unnamed, "and the rest by name")
        self.assertLess(len(unnamed), 13500, "but not every entry in full")

    def test_a_lookup_by_name_finds_exactly_that_position(self):
        # The tool is asked for a position, so everyday names count there and the loose word search is not needed.
        for query, key in (("missionary", "missionary"), ("69", "sixty-nine"), ("leap frog", "leap-frog"),
                           ("lotus", "lotus"), ("wheelbarrow", "wheelbarrow"), ("candle", "candle"),
                           ("full nelson", "full-nelson-standing"), ("bullseye", "bullseye")):
            self.assertEqual([p["key"] for p in pose_guide.lookup(query)["positions"]], [key], query)

    def test_an_image_prompt_gets_the_geometry_of_the_position_it_names(self):
        note = pose_guide.image_note("Explicit photo of a couple in the mating press, hotel room")
        self.assertIn("Mating press:", note)
        self.assertIn("Camera:", note)
        self.assertEqual(pose_guide.image_note("A woman with a butterfly tattoo at a candle-lit dinner"), "")
        sentence = pose_guide.lookup("mating press")["positions"][0]["guide"].split("prompt: ")[1]
        self.assertEqual(pose_guide.image_note(f"Mating press. {sentence}"), "", "not repeated when already written out")

    def test_everyday_words_are_not_positions(self):
        for brief in ("She has a pink butterfly tattoo on her thigh", "A candle-lit dinner on the terrace",
                      "A cowgirl at a rodeo", "He walks the doggy in the park", "The wrestler hits a piledriver",
                      "Dick Grayson patrols Gotham", "The rooster's cock crows", "Route 69", "A pretzel stand", "Hit the bullseye",
                      "She watches Northern Exposure", "He draws the king of clubs", "A ham sandwich", "A wishbone at dinner"):
            self.assertEqual(pose_guide.planner_note(brief), "", brief)

    def test_in_a_sexual_brief_they_count_unless_they_name_a_thing(self):
        keys = lambda text: [p.key for p in pose_guide.mentioned(text)]
        self.assertEqual(keys("they have sex, candle position"), ["candle"])
        self.assertEqual(keys("prone bone in a candle-lit room"), ["prone-bone"])
        self.assertEqual(keys("sex, her butterfly tattoo showing, doggy"), ["doggy"])
        self.assertEqual(keys("mating press then piledriver"), ["mating-press", "piledriver"],
                         "a name nothing else means makes the brief sexual")

    def test_the_new_positions_are_found_by_the_names_people_use(self):
        keys = lambda text: [p.key for p in pose_guide.mentioned(text)]
        for brief, key in (("standing congress", "standing-face-to-face"), ("upstanding citizen, sex", "wall-lift"),
                           ("elevated missionary", "missionary-pillow"), ("sex in the bellarina", "ballerina"),
                           ("lotus face to face, sex", "lotus"), ("Seated Wheelbarrow", "seated-wheelbarrow"),
                           ("cowgirl helper", "cowgirls-helper"), ("rider on fire", "rider-on-fire"),
                           ("69, oral sex", "sixty-nine"), ("Face-Sitting", "face-sitting"),
                           ("Butter Churner", "butter-churner"), ("Stand And Deliver, sex", "stand-and-deliver"),
                           ("standing rear entry", "standing-doggy"), ("straddle on all four", "straddle-all-fours"),
                           ("bullseye, oral sex", "bullseye"), ("DP Cowgirl", "dp-cowgirl"),
                           ("Doggy Double Penetration", "dp-doggy"), ("Missionary Double, threesome", "dp-missionary"),
                           ("Reverse Right Angle, double penetration", "dp-reverse-right-angle"),
                           ("Piledriver DP", "dp-piledriver"), ("spooning dp", "dp-spooning")):
            self.assertEqual(keys(brief), [key], brief)


if __name__ == "__main__":
    unittest.main()
