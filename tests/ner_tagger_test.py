import unittest

from pandas import Series

from src.preprocessing import Preprocessing


class NERTaggerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Hey")

    @classmethod
    def tearDownClass(cls):
        print("ho")

    def test_tagging(self):
        preprocessing = Preprocessing()
        text_to_tag = "Angela Merkel ist die deutsche Bundeskanzlerin und sie ist Mitglied der CDU."

        series_persons = preprocessing._tag_persons(Series(text_to_tag))
        series_organizations = preprocessing._tag_organizations(Series(text_to_tag))

        self.assertEqual(
            series_persons.tolist(),
            ["Angela Merkel"],
            "The person list should contain Angela Merkel",
        )

        self.assertEqual(series_organizations.tolist(), ["CDU"], "The organization list should contain CDU")

    def test_filter_duplicates(self):
        preprocessing = Preprocessing()

        text_to_tag = (
            "Die FDP ist eine Partei. ",
            "Der Vorsitzende der FDP ist Christian Lindner. Die SPD ist eine andere Partei",
        )

        series_persons = preprocessing._tag_persons(Series(text_to_tag))
        series_organizations = preprocessing._tag_organizations(Series(text_to_tag))

        self.assertEqual(
            series_persons.tolist(),
            ["Christian Lindner"],
            "The person list should contain Christian Lindner",
        )

        self.assertEqual(
            series_organizations.tolist(),
            ["FDP", "SPD"],
            "FDP should be filtered out of the organization list",
        )


if __name__ == "__main__":
    unittest.main()
