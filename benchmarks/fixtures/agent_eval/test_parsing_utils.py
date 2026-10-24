import unittest

from parsing_utils import (
    file_extension,
    join_url,
    merge_headers,
    normalize_email,
    normalize_tags,
    optional_int,
    parse_bool,
    parse_duration_ms,
    redact_token,
)


class ParsingUtilsTest(unittest.TestCase):
    def test_parse_bool_understands_false_values(self):
        self.assertFalse(parse_bool("false"))
        self.assertTrue(parse_bool("YES"))

    def test_normalize_email_strips_whitespace(self):
        self.assertEqual(normalize_email(" User@Example.COM "), "user@example.com")

    def test_parse_duration_supports_ms_and_seconds(self):
        self.assertEqual(parse_duration_ms("250ms"), 250)
        self.assertEqual(parse_duration_ms("2s"), 2000)

    def test_join_url_preserves_base_path(self):
        self.assertEqual(join_url("https://api.test/v1/", "/jobs"), "https://api.test/v1/jobs")

    def test_normalize_tags_trims_and_removes_empty_values(self):
        self.assertEqual(normalize_tags(" Python, Agent, ,Redis "), ["python", "agent", "redis"])

    def test_file_extension_uses_final_suffix_lowercase(self):
        self.assertEqual(file_extension("archive.TAR.GZ"), "gz")

    def test_redact_token_handles_punctuation(self):
        self.assertEqual(redact_token("token=abc-123.x next"), "token=*** next")

    def test_merge_headers_is_case_insensitive(self):
        self.assertEqual(
            merge_headers({"Content-Type": "text/plain"}, {"content-type": "application/json"}),
            {"content-type": "application/json"},
        )

    def test_optional_int_preserves_missing_value(self):
        self.assertIsNone(optional_int(None))
        self.assertEqual(optional_int("42"), 42)


if __name__ == "__main__":
    unittest.main()
