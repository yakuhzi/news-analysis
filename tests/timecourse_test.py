import datetime
import os
import sys

testdir = os.path.dirname(__file__)
srcdir = "../src"
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

import unittest

from pandas import DataFrame

from src.time_course import TimeCourse, get_term_count_nouns, get_term_count_overall


class TimeCourseTest(unittest.TestCase):
    def test_get_term_count_nouns(self):
        df_paragraph = DataFrame(
            {
                "media": ["Tagesschau", "Tagesschau"],
                "party": ["CDU", "CDU"],
                "nouns": [["gesundheit", "gesundheit", "gesundheit", "finanzen"], ["schule"]],
                "date": ["2020-09-01", "2020-09-01"],
            }
        )

        weight = get_term_count_nouns(df_paragraph, "gesundheit")
        self.assertEqual(weight, 1.5)

    def test_get_term_count_overall(self):
        df_paragraph = DataFrame(
            {
                "media": ["Tagesschau", "Tagesschau"],
                "party": ["CDU", "CDU"],
                "text": [["gesundheit", "gesundheit", "gesundheit", "finanzen"], ["schule"]],
                "date": ["2020-09-01", "2020-09-01"],
            }
        )

        weight = get_term_count_overall(df_paragraph, "gesundheit")
        self.assertEqual(weight, 1.5)

    def test_configure_dataframe_for_time_course(self):
        start_date = datetime.datetime(2020, 8, 1, 0, 0)
        end_date = datetime.datetime(2020, 10, 1, 0, 0)
        df_paragraph = DataFrame(
            {
                "media": "Tagesschau",
                "party": "CDU",
                "nouns": ["gesundheit", "gesundheit"],
                "date": ["2020-09-01", "2020-11-01"],
            }
        )
        df_paragraph_test = DataFrame(
            {
                "media": "Tagesschau",
                "party": "CDU",
                "nouns": ["gesundheit", "gesundheit"],
                "date": ["2020-09-01", "2020-09-01"],
            }
        )
        time_course = TimeCourse()
        time_course.set_paragraph(df_paragraph)

        df_interval = time_course.configure_dataframe_for_time_course(start_date, end_date, "Tagesschau", df_paragraph)
        print(df_interval)
        print(df_paragraph_test)
        self.assertNotEqual(sorted(df_paragraph_test["nouns"].tolist()), sorted(df_interval["nouns"].tolist()))
        self.assertNotEqual(sorted(df_paragraph_test["media"].tolist()), sorted(df_interval["media"].tolist()))
