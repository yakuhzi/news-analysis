import unittest

from src.preprocessing_articles import PreprocessArticles


class NERTaggerTest(unittest.TestCase):
    def test_tagging(self):
        preprocess = PreprocessArticles()
        text_to_tag = "Angela Merkel ist die deutsche Bundeskanzlerin und sie ist Mitglied der CDU."
        person_list, organization_list = preprocess.tag(text_to_tag)
        self.assertEqual(
            person_list,
            ["Angela Merkel"],
            "The person list should contain Angela Merkel",
        )
        self.assertEqual(
            organization_list, ["CDU"], "The organization list should contain CDU"
        )

    def test_filter_duplicates(self):
        preprocess = PreprocessArticles()
        text_to_tag = (
            "Die FDP ist eine Partei. "
            "Der Vorsitzende der FDP ist Christian Lindner. Die SPD ist eine andere Partei"
        )
        person_list, organization_list = preprocess.tag(text_to_tag)
        self.assertEqual(
            person_list,
            ["Christian Lindner"],
            "The person list should contain Christian Lindner",
        )
        self.assertEqual(
            organization_list,
            ["FDP", "SPD"],
            "FDP should be filtered out of the organization list",
        )


if __name__ == "__main__":
    unittest.main()
