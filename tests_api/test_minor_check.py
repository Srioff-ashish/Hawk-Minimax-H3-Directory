"""The under-18 check: a character's own children don't make them a child, and nothing else gets through."""
import unittest

from hawk_api.manhwa import is_minor


def minor(text: str) -> bool:
    return is_minor("", {"appearance": text})


class FamilyChildrenAreNotTheCharacter(unittest.TestCase):
    def test_children_someone_has_or_hasnt_pass(self):
        for text in ("Raveen, 44. Married, no children, never smiles.",
                     "A 45 year old mother of two, her kids are grown",
                     "Widower raising three kids alone", "has two grown kids and a dog",
                     "their children left home years ago", "without kids, by choice", "childless, 50"):
            self.assertFalse(minor(text), text)

    def test_anything_describing_the_character_is_still_refused(self):
        for text in ("a child in the garden", "she looks like a kid", "the body of a child", "child-like frame",
                     "a child's face", "teenage girl", "17 year old", "a schoolgirl", "in a school uniform",
                     "no children, she is 16 years old", "her kids' friend, a teen", "little girl", "underage"):
            self.assertTrue(minor(text), text)


if __name__ == "__main__":
    unittest.main()
