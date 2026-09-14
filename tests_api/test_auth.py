"""Token checks and signed links. Pure Python."""

from __future__ import annotations

import os
import sys
import unittest
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hawk_api.auth import bearer, sign_path, signature_valid, split_path_token, token_matches  # noqa: E402

SECRET = "s3cret-token-for-tests-0123456789"


class Tokens(unittest.TestCase):
    def test_bearer_and_match(self):
        self.assertEqual(bearer("Bearer abc"), "abc")
        self.assertEqual(bearer("bearer  abc "), "abc")
        self.assertIsNone(bearer("Basic abc"))
        self.assertTrue(token_matches(SECRET, SECRET))
        self.assertFalse(token_matches(SECRET, SECRET[:-1]))
        self.assertFalse(token_matches(SECRET, None))

    def test_path_token(self):
        self.assertEqual(split_path_token(f"/t/{SECRET}/v1/jobs"), (SECRET, "/v1/jobs"))
        self.assertEqual(split_path_token(f"/t/{SECRET}/mcp"), (SECRET, "/mcp"))
        self.assertEqual(split_path_token("/v1/jobs"), (None, "/v1/jobs"))


class SignedLinks(unittest.TestCase):
    def parts(self, link):
        query = parse_qs(urlsplit(link).query)
        return urlsplit(link).path, query["exp"][0], query["sig"][0]

    def test_valid_expired_tampered(self):
        path, exp, sig = self.parts(sign_path(SECRET, "/v1/jobs/1/video", 60, now=1000))
        self.assertTrue(signature_valid(SECRET, path, exp, sig, now=1030))
        self.assertFalse(signature_valid(SECRET, path, exp, sig, now=1061))
        self.assertFalse(signature_valid(SECRET, "/v1/jobs/2/video", exp, sig, now=1030))
        self.assertFalse(signature_valid(SECRET, path, str(int(exp) + 100), sig, now=1030))
        self.assertFalse(signature_valid("other-secret-000000000", path, exp, sig, now=1030))
        self.assertFalse(signature_valid(SECRET, path, "soon", sig, now=1030))


if __name__ == "__main__":
    unittest.main()
