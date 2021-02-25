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
            ["für", "die", "cdu", "war", "das", "ein", "erfolg"],
            ["schlecht", "leistung", "der", "fdp"],
            ["negativ", "ergebnis", "der", "sitzung"],
            ["großartig", "erfolg", "nicht", "jedoch", "für", "die", "linke"],
            ["keine", "wertung", "für", "die", "spd"],
        ]

        polarity = [
            [0, 0, 0, 0, 0, 0, 0.5],
            [-0.5, 0, 0, 0],
            [-0.5, 0, 0, 0],
            [0.5, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]

        polarity_textblob = [0.5, -0.5, -0.5, 0.5, 0]

        df_paragraphs = DataFrame({"text": text, "polarity": polarity, "polarity_textblob": polarity_textblob})
        self.tfidf_sentiment = TfidfSentiment(df_paragraphs)

    def test_get_context_polarity(self):
        self.tfidf_sentiment.get_context_polarity(3)
        indices = self.tfidf_sentiment.df_paragraphs["party_indices"].tolist()
        polarity_3 = self.tfidf_sentiment.df_paragraphs["polarity_context"].tolist()

        self.tfidf_sentiment.get_context_polarity(5)
        polarity_5 = self.tfidf_sentiment.df_paragraphs["polarity_context"].tolist()

        self.assertEqual(indices, [[2], [3], [], [6], [4]])

        self.assertEqual(
            polarity_3,
            [
                [0, 0, 0, 0, 0, 0, 0],
                [-0.5, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        )

        self.assertEqual(
            polarity_5,
            [
                [0, 0, 0, 0, 0, 0, 0.5],
                [-0.5, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        )

    def test_calculate_sentiment_score(self):
        self.tfidf_sentiment.calculate_sentiment_score()
        sentiment_series = Series(
            [0.06850566426135879, -0.05233582502695436, -0.05233582502695436, 0.042455502543704976, 0.0]
        )
        output_sentiment_series = self.tfidf_sentiment.df_paragraphs["sentiment_score"]
        testing.assert_series_equal(output_sentiment_series, sentiment_series, check_names=False)

    def test_map_context_polarity(self):
        row = {"party_indices": [1, 5], "polarity": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

        polarity_1 = self.tfidf_sentiment._map_context_polarity(row, 1, 3)
        polarity_2 = self.tfidf_sentiment._map_context_polarity(row, 9, 3)
        polarity_3 = self.tfidf_sentiment._map_context_polarity(row, 1, 5)
        polarity_4 = self.tfidf_sentiment._map_context_polarity(row, 9, 5)

        self.assertEqual(polarity_1, 2)
        self.assertEqual(polarity_2, 0)
        self.assertEqual(polarity_3, 2)
        self.assertEqual(polarity_4, 10)

    def test_map_sentiment(self):
        self.tfidf_sentiment.get_context_polarity(3)
        self.tfidf_sentiment.calculate_sentiment_score()
        self.tfidf_sentiment.map_sentiment()

        sentiment_series = Series(["Positive", "Negative", "Negative", "Positive", "Neutral"])
        sentiment_context = Series(["Neutral", "Negative", "Neutral", "Neutral", "Neutral"])

        sentiment_series_sentiws = self.tfidf_sentiment.df_paragraphs["sentiment"]
        sentiment_series_textblob = self.tfidf_sentiment.df_paragraphs["sentiment_textblob"]
        sentiment_series_context = self.tfidf_sentiment.df_paragraphs["sentiment_context"]

        testing.assert_series_equal(sentiment_series_sentiws, sentiment_series, check_names=False)
        testing.assert_series_equal(sentiment_series_textblob, sentiment_series, check_names=False)
        testing.assert_series_equal(sentiment_series_context, sentiment_context, check_names=False)

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
