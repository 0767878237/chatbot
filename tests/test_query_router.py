from __future__ import annotations

import unittest

from rag.query_router import analyze_query, is_food_domain_query, is_greeting_query, normalize_text


class QueryRouterTests(unittest.TestCase):
    def test_normalize_text_keeps_vietnamese_meaning(self) -> None:
        self.assertEqual(normalize_text("món ngon ở TPHCM"), "mon ngon o tphcm")
        self.assertEqual(normalize_text("quán nào ở Bình Thạnh"), "quan nao o binh thanh")
        self.assertEqual(normalize_text("xin chào"), "xin chao")

    def test_greeting_query_detected_with_accents(self) -> None:
        analysis = analyze_query("xin chào")
        self.assertTrue(is_greeting_query(analysis.normalized_query))
        self.assertFalse(is_food_domain_query(analysis))

    def test_food_query_detected_with_accents(self) -> None:
        analysis = analyze_query("món ngon ở TPHCM")
        self.assertTrue(is_food_domain_query(analysis))
        self.assertIn("recommend", analysis.intents)

    def test_location_query_detected_with_accents(self) -> None:
        analysis = analyze_query("quán nào ở Bình Thạnh")
        self.assertTrue(is_food_domain_query(analysis))
        self.assertIn("binh thanh", analysis.location_terms)


if __name__ == "__main__":
    unittest.main()
