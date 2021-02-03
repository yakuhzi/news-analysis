import os
import sys

testdir = os.path.dirname(__file__)
srcdir = "../src"
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

import unittest

import spacy
from pandas import Series

from src.preprocessing import Preprocessing


class NERTaggerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = spacy.load("de_core_news_lg", disable=["parser"])
        cls.preprocessing = Preprocessing()

    def test_tagging(self):
        text_to_tag = "Angela Merkel ist die deutsche Bundeskanzlerin und sie ist Mitglied der CDU."
        series = Series([self.nlp(text_to_tag)])

        persons = self.preprocessing._tag_persons(series).tolist()[0]
        organizations = self.preprocessing._tag_organizations(series).tolist()[0]

        self.assertEqual(persons, ["Angela Merkel"], "The person list should contain Angela Merkel")
        self.assertEqual(organizations, ["CDU"], "The organization list should contain CDU")

    def test_filter_duplicates(self):
        text_to_tag = (
            "Die FDP ist eine Partei. Der Vorsitzende der FDP ist Christian Lindner. Die SPD ist eine " "andere Partei!"
        )
        series = Series([self.nlp(text_to_tag)])

        persons = self.preprocessing._tag_persons(series).tolist()[0]
        organizations = sorted(self.preprocessing._tag_organizations(series).tolist()[0])

        self.assertEqual(persons, ["Christian Lindner"], "The person list should contain Christian Lindner")
        self.assertEqual(organizations, ["FDP", "SPD"], "FDP should be filtered out of the organization list")


if __name__ == "__main__":
    unittest.main()
