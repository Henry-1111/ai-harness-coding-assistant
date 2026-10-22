import unittest

from text_utils import slugify


class TextUtilsTest(unittest.TestCase):
    def test_collapses_repeated_whitespace(self):
        self.assertEqual(slugify("LLM   Gateway"), "llm-gateway")

    def test_strips_surrounding_whitespace(self):
        self.assertEqual(slugify("  Safe Agent  "), "safe-agent")


if __name__ == "__main__":
    unittest.main()
