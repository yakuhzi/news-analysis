import os
import sys
import datetime
testdir = os.path.dirname(__file__)
srcdir = "../src"
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

import unittest

from pandas import DataFrame

from src.time_course import TimeCourse
from src.time_course import get_term_count_nouns
from src.time_course import get_term_count_overall

class TimeCourseTest(unittest.TestCase):

    def test_get_term_count_nouns(self):
        df_paragraph = DataFrame(
            {
                "media": "Tagesschau",
                "party": "CDU",
                "nouns": ["gesundheit", "gesundheit", "gesundheit", "Hallo"],
                "date": "2020-09-01"
            })

        weight = get_term_count_nouns(df_paragraph, "gesundheit")
        self.assertEqual(weight, 3)

    def test_get_term_count_overall(self):
        df_paragraph = DataFrame(
            {
                "media": "Tagesschau",
                "party": "CDU",
                "text": ["gesundheit", "gesundheit", "gesundheit", "Hallo"],
                "date": "2020-09-01"
            })

        weight = get_term_count_overall(df_paragraph, "gesundheit")
        self.assertEqual(weight, 3)

    def test_configure_dataframe_for_time_course(self):
        start_date = datetime.datetime(2020, 8, 1, 0, 0)
        end_date = datetime.datetime(2020, 10, 1, 0, 0)
        df_paragraph = DataFrame(
            {
                "media": "Tagesschau",
                "party": "CDU",
                "nouns": ["gesundheit", "gesundheit"],
                "date": ["2020-09-01", "2020-11-01"]
            }
        )
        df_paragraph_test = DataFrame(
            {
                "media": "Tagesschau",
                "party": "CDU",
                "nouns": ["gesundheit", "gesundheit"],
                "date": ["2020-09-01", "2020-09-01"]
            })
        time_course = TimeCourse()
        time_course.set_paragraph(df_paragraph)

        df_interval = time_course.configure_dataframe_for_time_course(start_date, end_date, "Tagesschau", df_paragraph)
        print(df_interval)
        print(df_paragraph_test)
        self.assertNotEqual(sorted(df_paragraph_test["nouns"].tolist()), sorted(df_interval["nouns"].tolist()))
        self.assertNotEqual(sorted(df_paragraph_test["media"].tolist()), sorted(df_interval["media"].tolist()))

    def test_frequency_count(self):
        party_list = ["CDU"]
        media_list = ["Tagesschau"]
        df_top_terms = DataFrame({
            "party": "CDU",
            "term": "gesundheit"
        }, index=[0])
        initial_start_date = datetime.datetime(2020, 8, 1, 0, 0)
        initial_end_date = datetime.datetime(2020, 10, 1, 0, 0)
        df_paragraph = DataFrame(
            {
                "media": "Tagesschau",
                "party": "CDU",
                "nouns": ["gesundheit", "gesundheit", "Hallo ", "gesundheit"],
                "date": "2020-09-01"
            }
        )

        df_images = DataFrame({
            "party": "CDU",
            "media": "Tagesschau",
            "term": "gesundheit",
            "weight": [3]
        })

        time_course = TimeCourse()
        images = time_course.get_time_course(party_list, media_list, df_top_terms, initial_start_date, initial_end_date,
                                             df_paragraph)
        #self.assertEqual(sorted(images["party"].tolist()), sorted(df_images["party"].tolist()))
        #self.assertEqual(sorted(images["media"].tolist()), sorted(df_images["media"].tolist()))
        #self.assertEqual(sorted(images["weight"].tolist()), sorted(df_images["weight"].tolist()))



