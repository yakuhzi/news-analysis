from typing import Tuple

import numpy as np
from pandas import Series
from sklearn import metrics

from tfidf_sentiment import TfidfSentiment
from utils.reader import Reader


class Comparison:
    def __init__(self, filename: str):
        self.dataframe = Reader.read_json_to_df_default("src/output/" + filename + ".json")

    def train_threshold(self):
        tfidf_sentiment = TfidfSentiment(self.dataframe)
        t = 0
        best_threshold = 0
        best_score = 0

        while t <= 0.1:
            tfidf_sentiment.map_sentiment(threshold=t, overwrite=True)
            self.dataframe = tfidf_sentiment.df_paragraphs

            f1_sentiws, _ = self.f1_score()
            f1_postive_negative = f1_sentiws[0] + f1_sentiws[1] + f1_sentiws[2]

            if f1_postive_negative > best_score:
                best_score = f1_postive_negative
                best_threshold = t

            t += 0.00001

        tfidf_sentiment.map_sentiment(threshold=best_threshold, overwrite=True)
        return best_threshold, best_score

    def precision(self) -> Tuple[float, float]:
        labels = ["Positive", "Negative", "Neutral"]
        precision_sentiws = metrics.precision_score(
            self.dataframe["labeled_sentiment"], self.dataframe["sentiment"], labels=labels, average=None
        )
        precision_textblob = metrics.precision_score(
            self.dataframe["labeled_sentiment"], self.dataframe["sentiment_textblob"], labels=labels, average=None
        )
        return precision_sentiws, precision_textblob

    def recall(self):
        labels = ["Positive", "Negative", "Neutral"]
        recall_sentiws = metrics.recall_score(
            self.dataframe["labeled_sentiment"], self.dataframe["sentiment"], labels=labels, average=None
        )
        recall_textblob = metrics.recall_score(
            self.dataframe["labeled_sentiment"], self.dataframe["sentiment_textblob"], labels=labels, average=None
        )
        return recall_sentiws, recall_textblob

    def accuracy(self):
        accuracy_sentiws = metrics.accuracy_score(self.dataframe["labeled_sentiment"], self.dataframe["sentiment"])
        accuracy_textblob = metrics.accuracy_score(
            self.dataframe["labeled_sentiment"], self.dataframe["sentiment_textblob"]
        )
        return accuracy_sentiws, accuracy_textblob

    def f1_score(self):
        labels = ["Positive", "Negative", "Neutral"]
        f1_sentiws = metrics.f1_score(
            self.dataframe["labeled_sentiment"], self.dataframe["sentiment"], labels=labels, average=None
        )
        f1_textblob = metrics.f1_score(
            self.dataframe["labeled_sentiment"], self.dataframe["sentiment_textblob"], labels=labels, average=None
        )
        return f1_sentiws, f1_textblob

    def polarity(self):
        unlabeled_series = self.dataframe["sentiment"]
        labeled_series = self.dataframe["labeled_sentiment"]
        unlabeled_series_textblob = self.dataframe["sentiment_textblob"]

        # sentiWS
        comparison = np.where((unlabeled_series == labeled_series), True, False)
        number_of_equal = np.count_nonzero(comparison)

        # TextBlob
        comparison_textblob = np.where((unlabeled_series_textblob == labeled_series), True, False)
        number_of_equal_textblob = np.count_nonzero(comparison_textblob)

        # SentiWS
        comparison_neutral = np.where(
            ((unlabeled_series == labeled_series) & (unlabeled_series == "Neutral")), True, False
        )
        number_of_equal_neutral = np.count_nonzero(comparison_neutral)
        neutral = np.where(labeled_series == "Neutral", True, False)
        number_of_neutrals = np.count_nonzero(neutral)

        # TextBlob
        comparison_neutral_textblob = np.where(
            ((unlabeled_series_textblob == labeled_series) & (unlabeled_series_textblob == "Neutral")), True, False
        )
        number_of_equal_neutral_textblob = np.count_nonzero(comparison_neutral_textblob)
        neutral_textblob = np.where(labeled_series == "Neutral", True, False)
        number_of_neutrals_textblob = np.count_nonzero(neutral_textblob)

        # SentiWS
        comparison_positive = np.where(
            ((unlabeled_series == labeled_series) & (unlabeled_series == "Positive")), True, False
        )
        number_of_equal_positives = np.count_nonzero(comparison_positive)
        positive = np.where(labeled_series == "Positive", True, False)
        number_of_positives = np.count_nonzero(positive)

        # TextBlob
        comparison_positive_textblob = np.where(
            ((unlabeled_series_textblob == labeled_series) & (unlabeled_series_textblob == "Positive")), True, False
        )
        number_of_equal_positives_textblob = np.count_nonzero(comparison_positive_textblob)
        positive_textblob = np.where(labeled_series == "Positive", True, False)
        number_of_positives_textblob = np.count_nonzero(positive_textblob)

        # SentiWS
        comparison_negative = np.where(
            ((unlabeled_series == labeled_series) & (unlabeled_series == "Negative")), True, False
        )
        number_of_equal_negatives = np.count_nonzero(comparison_negative)
        negative = np.where(labeled_series == "Negative", True, False)
        number_of_negatives = np.count_nonzero(negative)

        # TextBlob
        comparison_negative_textblob = np.where(
            ((unlabeled_series_textblob == labeled_series) & (unlabeled_series_textblob == "Negative")), True, False
        )
        number_of_equal_negatives_textblob = np.count_nonzero(comparison_negative_textblob)
        negative_textblob = np.where(labeled_series == "Negative", True, False)
        number_of_negatives_textblob = np.count_nonzero(negative_textblob)

        print("Polarity matched with labeled data: {} out of {} times".format(number_of_equal, len(comparison)))
        print("Neutral polarity matched: {} out of {} times".format(number_of_equal_neutral, number_of_neutrals))
        print("Positive polarity matched: {} out of {} times".format(number_of_equal_positives, number_of_positives))
        print("Negative polarity matched: {} out of {} times".format(number_of_equal_negatives, number_of_negatives))

        print("Unlabeled {}: {}".format(unlabeled_series.value_counts().keys()[0], unlabeled_series.value_counts()[0]))
        print("Unlabeled {}: {}".format(unlabeled_series.value_counts().keys()[1], unlabeled_series.value_counts()[1]))
        print("Unlabeled {}: {}".format(unlabeled_series.value_counts().keys()[2], unlabeled_series.value_counts()[2]))

        print("Labeled {}: {}".format(labeled_series.value_counts().keys()[0], labeled_series.value_counts()[0]))
        print("Labeled {}: {}".format(labeled_series.value_counts().keys()[1], labeled_series.value_counts()[1]))
        print("Labeled {}: {}".format(labeled_series.value_counts().keys()[2], labeled_series.value_counts()[2]))

        # compare textblob
        print(
            "==============================TextBlob================================Comparison=========================="
        )
        print(
            "Polarity of Textblob matched with labeled data: {} out of {} times".format(
                number_of_equal_textblob, len(comparison_textblob)
            )
        )
        print(
            "Neutral polarity matched: {} out of {} times".format(
                number_of_equal_neutral_textblob, number_of_neutrals_textblob
            )
        )
        print(
            "Positive polarity matched: {} out of {} times".format(
                number_of_equal_positives_textblob, number_of_positives_textblob
            )
        )
        print(
            "Negative polarity matched: {} out of {} times".format(
                number_of_equal_negatives_textblob, number_of_negatives_textblob
            )
        )

    def polarity_to_subjectivity(self):
        polarity_series = self.dataframe["labeled_sentiment"]
        subjectivity_series = self.dataframe["labeled_subjectivity"]

        not_neutral_when_subjective = np.where(
            (polarity_series != "Neutral") & (subjectivity_series == "Subjective"), True, False
        )
        not_neutral_when_objective = np.where(
            (polarity_series != "Neutral") & (subjectivity_series == "Objective"), True, False
        )

        subjective_series = Series(not_neutral_when_subjective)
        objective_series = Series(not_neutral_when_objective)

        print("==========================================================================")
        print("Not neutral but subjective: {}".format(subjective_series.value_counts()[1]))
        print("Not neutral but objective: {}".format(objective_series.value_counts()[1]))
