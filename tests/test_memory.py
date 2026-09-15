"""Out-of-memory retry helper. Pure Python: python -m unittest discover -s tests"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawk_h3.memory import run_with_oom_retry  # noqa: E402


class FakeOOM(Exception):
    pass


class Harness:
    def __init__(self, failures: int, error: type = FakeOOM):
        self.failures, self.error = failures, error
        self.calls = self.recovers = 0
        self.logs: list[str] = []

    def fn(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error("boom")
        return "frames"

    def run(self):
        return run_with_oom_retry(
            self.fn,
            label="video decode",
            is_oom=lambda exc: isinstance(exc, FakeOOM),
            recover=self.recover,
            log=self.logs.append,
        )

    def recover(self):
        self.recovers += 1


class OomRetry(unittest.TestCase):
    def test_success_runs_once(self):
        harness = Harness(failures=0)
        self.assertEqual(harness.run(), "frames")
        self.assertEqual((harness.calls, harness.recovers, harness.logs), (1, 0, []))

    def test_one_oom_recovers_and_retries(self):
        harness = Harness(failures=1)
        self.assertEqual(harness.run(), "frames")
        self.assertEqual((harness.calls, harness.recovers), (2, 1))
        self.assertIn("video decode", harness.logs[0])

    def test_second_oom_explains_what_to_lower(self):
        harness = Harness(failures=2)
        with self.assertRaisesRegex(RuntimeError, "video decode even after unloading") as ctx:
            harness.run()
        self.assertIsInstance(ctx.exception.__cause__, FakeOOM)
        self.assertEqual((harness.calls, harness.recovers), (2, 1))
        self.assertIn("retry the job to resume", str(ctx.exception))

    def test_other_errors_pass_through(self):
        harness = Harness(failures=1, error=ValueError)
        with self.assertRaises(ValueError):
            harness.run()
        self.assertEqual((harness.calls, harness.recovers), (1, 0))

    def test_non_oom_on_retry_passes_through(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise FakeOOM() if calls["n"] == 1 else KeyError("later")

        with self.assertRaises(KeyError):
            run_with_oom_retry(fn, label="x", is_oom=lambda e: isinstance(e, FakeOOM), recover=lambda: None, log=lambda m: None)


if __name__ == "__main__":
    unittest.main()
