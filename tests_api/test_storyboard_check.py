"""The panel check keeps panels whose only fault is a person with no identity picture (the reasons below are the ones
a live pod gave for dropping 11 of 12 panels on one board), and still drops real defects."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hawk_api.storyboard import CHECK_PROMPT, only_unmatched_people


class OnlyUnmatched(unittest.TestCase):
    def test_reasons_seen_live_are_not_defects(self):
        for issues in ("unmatched man (no identity reference)", "two men match none of the identity references",
                       "unidentified man matching no identity reference", "unrecognized people not matching any identity reference",
                       "unmatched men (no identity refs)", "unidentified man (no matching identity ref)",
                       "unrecognized person not matching any identity reference"):
            with self.subTest(issues=issues):
                self.assertTrue(only_unmatched_people(issues))

    def test_real_defects_are_still_defects(self):
        for issues in ("broken anatomy (male chest fused to female buttocks); unmatched persons",
                       "extra fingers on the left hand", "garbled text on the sign", "her face looks different from the reference",
                       "two men match none of the identity references; woman's breast size/build does not match refs",
                       "watermark in the corner"):
            with self.subTest(issues=issues):
                self.assertFalse(only_unmatched_people(issues))

    def test_the_prompt_says_people_without_a_picture_never_fail_a_panel(self):
        self.assertIn("NEVER fail a panel because someone in it matches no", CHECK_PROMPT)
        self.assertNotIn("matches NONE of the identity references", CHECK_PROMPT)


if __name__ == "__main__":
    unittest.main()
