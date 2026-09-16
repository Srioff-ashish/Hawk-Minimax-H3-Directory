"""Music bed mixing. Needs torch (skipped without it):
    python -m unittest tests.test_media
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch

    from hawk_h3 import media
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest(f"torch missing: {exc}")


class MusicBed(unittest.TestCase):
    RATE = 1000

    def test_loops_to_film_length_and_fades_out(self):
        scene = torch.zeros(1, 2, 5 * self.RATE)
        music = torch.full((1, 1, 2 * self.RATE), 0.5)
        mixed = media.mix_music_bed(scene, self.RATE, music, self.RATE, music_db=0.0, fade_seconds=1.0)
        self.assertEqual(tuple(mixed.shape), (1, 2, 5 * self.RATE))
        self.assertAlmostEqual(float(mixed[0, 0, 0]), 0.0, places=6)  # fade-in starts silent
        self.assertAlmostEqual(float(mixed[0, 1, 3 * self.RATE]), 0.5, places=5)  # looped, both channels
        self.assertAlmostEqual(float(mixed[0, 0, -1]), 0.0, places=6)  # faded out
        self.assertAlmostEqual(float(mixed[0, 0, int(4.5 * self.RATE)]), 0.25, delta=0.01)

    def test_trims_resamples_and_applies_levels(self):
        scene = torch.full((1, 1, 2 * self.RATE), 0.2)
        music = torch.full((1, 2, 10 * 2000), 0.2)
        mixed = media.mix_music_bed(scene, self.RATE, music, 2000, music_db=-6.0, scene_db=-6.0, fade_seconds=0.0)
        self.assertEqual(tuple(mixed.shape), (1, 1, 2 * self.RATE))
        gain = media.db_to_gain(-6.0)
        self.assertAlmostEqual(float(mixed[0, 0, self.RATE]), 0.2 * gain * 2, places=3)

    def test_limits_peaks(self):
        scene = torch.full((1, 1, self.RATE), 0.9)
        music = torch.full((1, 1, self.RATE), 0.9)
        mixed = media.mix_music_bed(scene, self.RATE, music, self.RATE, music_db=0.0, fade_seconds=0.0)
        self.assertLessEqual(float(mixed.abs().max()), 0.99 + 1e-6)


if __name__ == "__main__":
    unittest.main()
