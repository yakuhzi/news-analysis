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

        df_paragraphs = DataFrame({"nouns": nouns, "parties": cls.parties, "media": media})

        cls.keyword_extraction = KeywordExtraction(df_paragraphs)

    def test_get_term_weight_tuples(self):
        tuples = self.keyword_extraction.get_term_weight_tuples(parties=["CDU", "Grüne"], topn=1)

        df_tuples = DataFrame(
            {
                "party": ["CDU", "CDU", "Grüne", "Grüne"],
                "term": ["gesundheit", "umwelt", "gesundheit", "umwelt"],
                "weight": [6, 1, 1, 6],
            }
        )

        testing.assert_frame_equal(tuples, df_tuples, check_like=True)


if __name__ == "__main__":
    unittest.main()
