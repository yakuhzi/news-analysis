import os
import sys

testdir = os.path.dirname(__file__)
srcdir = "../src"
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

import unittest

from pandas import DataFrame, testing

from src.keyword_extraction import KeywordExtraction


class KeywordExtractionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.party_nouns = ["gesundheit", "corona", "studium", "finanzen", "schule", "umwelt", "polizei"]
        common_nouns = ["partei", "artikel", "justiz"] + cls.party_nouns

        nouns = [
            common_nouns + ["gesundheit", "gesundheit", "gesundheit", "gesundheit", "gesundheit"],
            common_nouns + ["corona", "corona", "corona", "corona", "corona"],
            common_nouns + ["studium", "studium", "studium", "studium", "studium"],
            common_nouns + ["finanzen", "finanzen", "finanzen", "finanzen", "finanzen"],
            common_nouns + ["schule", "schule", "schule", "schule", "schule"],
            common_nouns + ["umwelt", "umwelt", "umwelt", "umwelt", "umwelt"],
            common_nouns + ["polizei", "polizei", "polizei", "polizei", "polizei"],
        ]

        cls.parties = [["CDU"], ["CSU"], ["SPD"], ["FDP"], ["AfD"], ["Grüne"], ["Linke"]]
        media = ["Bild", "Tagesschau", "TAZ", "Bild", "TAZ", "Tagesschau", "Tagesschau"]

        cls.df_paragraphs = DataFrame({"nouns": nouns, "parties": cls.parties, "media": media})
        cls.keyword_extraction = KeywordExtraction(cls.df_paragraphs)

    def test_get_term_weight_tuples_by_party(self):
        tuples = self.keyword_extraction.get_term_weight_tuples(by_party=True, parties=["CDU", "Grüne"], topn=1)

        df_tuples = DataFrame(
            {
                "party": ["CDU", "CDU", "Grüne", "Grüne"],
                "term": ["gesundheit", "umwelt", "gesundheit", "umwelt"],
                "weight": [6, 1, 1, 6],
            }
        )

        self.assertEqual(sorted(tuples["party"].tolist()), sorted(df_tuples["party"].tolist()))
        self.assertEqual(sorted(tuples["term"].tolist()), sorted(df_tuples["term"].tolist()))
        self.assertEqual(sorted(tuples["weight"].tolist()), sorted(df_tuples["weight"].tolist()))

    def test_get_term_weight_tuples_by_media(self):
        tuples = self.keyword_extraction.get_term_weight_tuples(by_party=False, media=["TAZ", "Bild"], topn=1)

        df_tuples = DataFrame(
            {
                "media": ["TAZ", "TAZ", "Bild", "Bild"],
                "term": ["schule", "finanzen", "schule", "finanzen"],
                "weight": [7, 2, 2, 7],
            }
        )

        self.assertEqual(sorted(tuples["media"].tolist()), sorted(df_tuples["media"].tolist()))
        self.assertEqual(sorted(tuples["term"].tolist()), sorted(df_tuples["term"].tolist()))
        self.assertEqual(sorted(tuples["weight"].tolist()), sorted(df_tuples["weight"].tolist()))

    def test_get_term_count_by_party(self):
        term_count = self.keyword_extraction.get_term_count(self.df_paragraphs, True, "CDU", "gesundheit")
        self.assertEqual(term_count, 6)

    def test_get_term_count_by_media(self):
        term_count = self.keyword_extraction.get_term_count(self.df_paragraphs, False, "Tagesschau", "gesundheit")
        self.assertEqual(term_count, 3)


if __name__ == "__main__":
    unittest.main()
