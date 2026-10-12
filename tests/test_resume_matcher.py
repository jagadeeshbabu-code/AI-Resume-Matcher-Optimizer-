import unittest

from resume_matcher import (
    build_suggestions,
    find_missing_keywords,
    normalize_text,
)


class ResumeMatcherTests(unittest.TestCase):
    def test_normalize_text_is_case_insensitive_and_collapses_punctuation(self):
        self.assertEqual(normalize_text("Python,  C++ & Machine-Learning!"), "python c++ machine learning")

    def test_find_missing_keywords_uses_phrase_boundaries(self):
        matched, missing = find_missing_keywords(
            ["python", "project management", "java"],
            "Experienced in Python and project management.",
        )
        self.assertEqual(matched, ["python", "project management"])
        self.assertEqual(missing, ["java"])

    def test_suggestions_are_constructive_and_limited(self):
        suggestions = build_suggestions(["Python", "SQL"])
        self.assertEqual(len(suggestions), 2)
        self.assertIn("Python", suggestions[0])
        self.assertIn("If accurate", suggestions[0])

    def test_empty_missing_list_returns_positive_review_guidance(self):
        suggestions = build_suggestions([])
        self.assertEqual(len(suggestions), 1)
        self.assertIn("all of the extracted", suggestions[0])


if __name__ == "__main__":
    unittest.main()
