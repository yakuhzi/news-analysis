import os
import sys

testdir = os.path.dirname(__file__)
srcdir = "../src"
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

import unittest

from pandas import DataFrame, Series, testing

from src.tfidf_sentiment import TfidfSentiment


class TfidfSentimentTest(unittest.TestCase):
    def setUp(self):
        text = [
            ["das", "ist", "ein", "guter", "tag"],
            ["das", "ist", "ein", "schlechter", "tag"],
            ["das", "ist", "ein", "normaler", "tag"],
        ]

        polarity = [
            [0, 0, 0, 0.4, 0],
            [0, 0, 0, -0.4, 0],
            [0, 0, 0, 0, 0],
        ]

        polarity_textblob = [0.4, -0.4, 0]

        df_paragraphs = DataFrame({"text": text, "polarity": polarity, "polarity_textblob": polarity_textblob})

        self.tfidf_sentiment = TfidfSentiment(df_paragraphs)

    def test_calculate_sentiment_score(self):
        self.tfidf_sentiment.calculate_sentiment_score()
        sentiment_series = Series([0.08615052200619643, -0.08615052200619643, 0.0])
        output_sentiment_series = self.tfidf_sentiment.df_paragraphs["sentiment_score"]
        testing.assert_series_equal(output_sentiment_series, sentiment_series, check_names=False)

    def test_map_sentiment(self):
        self.tfidf_sentiment.calculate_sentiment_score()
        self.tfidf_sentiment.map_sentiment()
        sentiment_series = Series(["Positive", "Negative", "Neutral"])

        sentiment_series_sentiws = self.tfidf_sentiment.df_paragraphs["sentiment"]
        sentiment_series_textblob = self.tfidf_sentiment.df_paragraphs["sentiment_textblob"]

        testing.assert_series_equal(sentiment_series_sentiws, sentiment_series, check_names=False)
        testing.assert_series_equal(sentiment_series_textblob, sentiment_series, check_names=False)

    def test_map_sentiment_row(self):
        sentiment_positive = self.tfidf_sentiment._map_sentiment("0.1")
        sentiment_negative = self.tfidf_sentiment._map_sentiment("-0.1")
        sentiment_neutral = self.tfidf_sentiment._map_sentiment("0")

        self.assertEqual(sentiment_positive, "Positive")
        self.assertEqual(sentiment_negative, "Negative")
        self.assertEqual(sentiment_neutral, "Neutral")

    def test_remove_umlauts(self):
        text_with_umlauts = (
            "Heiße Suppe schmeckt mir überaus gut und könnte ich ruhig öfter genießen. "
            "Hätte ich mir doch mehr gemacht."
        )

        text_without_umlauts = self.tfidf_sentiment._remove_umlauts(text_with_umlauts)
        self.assertEqual(
            text_without_umlauts,
            "Heisse Suppe schmeckt mir ueberaus gut und koennte ich ruhig oefter "
            "geniessen. Haette ich mir doch mehr gemacht.",
        )


if __name__ == "__main__":
    unittest.main()
