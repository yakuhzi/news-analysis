from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series
from sklearn import metrics

from tfidf_sentiment import TfidfSentiment
from utils.reader import Reader


class Comparison:
    def __init__(self, filename: str):
        self.dataframe = Reader.read_json_to_df_default("src/output/" + filename + ".json")

    def train_threshold(self) -> float:
        """
        train the threshold with labeled data
        :return the best threshold
        """
        tfidf_sentiment = TfidfSentiment(self.dataframe)
        t = 0
        best_threshold = 0
        best_score = 0
        thresholds = []
        f1_positive = []
        f1_negative = []
        f1_neutral = []
        # iterate over different thresholds, increase with every loop
        while t <= 0.005:
            tfidf_sentiment.map_sentiment(threshold=t, overwrite=True)
            self.dataframe = tfidf_sentiment.df_paragraphs
            # optimize over the sum of all f1 scores for sentiws
            f1_sentiws, _ = self.f1_score(training=True)
            f1_postive_negative = f1_sentiws[0] + f1_sentiws[1] + f1_sentiws[2]

            thresholds.append(t)
            f1_positive.append(f1_sentiws[0])
            f1_negative.append(f1_sentiws[1])
            f1_neutral.append(f1_sentiws[2])
            # replace best threshold if current one is better
            if f1_postive_negative > best_score:
                best_score = f1_postive_negative
                best_threshold = t

            t += 0.0000001

        # visualize the training
        self.visualize_threshold(thresholds, f1_positive, f1_negative, f1_neutral, best_threshold)
        # adjust the sentiment
        tfidf_sentiment.map_sentiment(threshold=best_threshold, overwrite=True)
        return best_threshold

    def visualize_threshold(
        self,
        threshold: List[float],
        f1_positive: List[float],
        f1_negative: List[float],
        f1_neutral: List[float],
        best_threshold: float,
    ) -> None:
        """
        visualize the results of training the threshold
        :param threshold: all thresholds used for training as list
        :param f1_positive: the f1 scores for positive labeled data
        :param f1_negative: the f1 scores for negative labeled data
        :param f1_neutral: the f1 scores for neutral labeled data
        :param best_threshold: the best threshold from training
        """
        plt.axis((0, 0.005, 0, 1))
        # plot f1 scores as line graphs
        plt.plot(threshold, f1_positive, color="green", label="f1 positive")
        plt.plot(threshold, f1_negative, color="orange", label="f1 negative")
        plt.plot(threshold, f1_neutral, color="blue", label="f1 neutral")
        # plot the best found threshold as vertical line
        plt.axvline(best_threshold, color="red", label="best threshold")
        plt.legend()
        plt.show()

    def _output_metric(
        self,
        metric: str,
        column_sentiws: str,
        column_textblob: str,
        result_sentiws: np.ndarray,
        result_textblob: np.ndarray,
    ) -> None:
        """
        print the result of metrics calculation as pandas data frame
        :param metric: the metric to print
        :param column_sentiws: the column name for the sentiws result
        :param column_textblob: the column name for the textblob result
        :param result_sentiws: the resulting metric for sentiws
        :param result_textblob: the resulting metric for textblob
        """
        data = {
            column_sentiws: [result_sentiws[0], result_sentiws[1], result_sentiws[2]],
            column_textblob: [result_textblob[0], result_textblob[1], result_textblob[2]],
        }
        df = pd.DataFrame(data=data, index=["Positive", "Negative", "Neutral"])
        print("==================== " + metric + " ====================\n")
        print(df.to_markdown() + "\n\n")

    def precision(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        calculate the precision of the labeled data for sentiws and textblob
        :return: a tuple of the precision scores for positive, negative and neutral labeled data
                 for sentiws and textblob
        """
        labels = ["Positive", "Negative", "Neutral"]

        # calculate precision for sentiws
        precision_sentiws = metrics.precision_score(
            self.dataframe["labeled_sentiment"], self.dataframe["sentiment"], labels=labels, average=None
        )

        # calculate precision for textblob
        precision_textblob = metrics.precision_score(
            self.dataframe["labeled_sentiment"], self.dataframe["sentiment_textblob"], labels=labels, average=None
        )

        # print the result
        self._output_metric(
            metric="Precision",
            column_sentiws="precision_sentiws",
            column_textblob="precision_textblob",
            result_sentiws=precision_sentiws,
            result_textblob=precision_textblob,
        )

        return precision_sentiws, precision_textblob

    def recall(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        calculate the recall of the labeled data for sentiws and textblob
        :return: a tuple of the recall scores for positive, negative and neutral labeled data
                 for sentiws and textblob
        """
        labels = ["Positive", "Negative", "Neutral"]

        # calculate recall for sentiws
        recall_sentiws = metrics.recall_score(
            self.dataframe["labeled_sentiment"], self.dataframe["sentiment"], labels=labels, average=None
        )

        # calculate recall for textblob
        recall_textblob = metrics.recall_score(
            self.dataframe["labeled_sentiment"], self.dataframe["sentiment_textblob"], labels=labels, average=None
        )

        # print the result
        self._output_metric(
            metric="Recall",
            column_sentiws="recall_sentiws",
            column_textblob="recall_textblob",
            result_sentiws=recall_sentiws,
            result_textblob=recall_textblob,
        )

        return recall_sentiws, recall_textblob

    def f1_score(self, training: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        calculate the f1 score of the labeled data for sentiws and textblob
        :param training: true if threshold is trained (do not print score)
        :return: a tuple of the f1 scores for positive, negative and neutral labeled data
                 for sentiws and textblob
        """
        labels = ["Positive", "Negative", "Neutral"]

        # calculate f1 score for sentiws
        f1_sentiws = metrics.f1_score(
            self.dataframe["labeled_sentiment"], self.dataframe["sentiment"], labels=labels, average=None
        )

        # calculate f1 score for textblob
        f1_textblob = metrics.f1_score(
            self.dataframe["labeled_sentiment"], self.dataframe["sentiment_textblob"], labels=labels, average=None
        )

        # print the result (not if threshold is trained)
        if not training:
            self._output_metric(
                metric="F1-score",
                column_sentiws="f1_sentiws",
                column_textblob="f1_textblob",
                result_sentiws=f1_sentiws,
                result_textblob=f1_textblob,
            )

        return f1_sentiws, f1_textblob

    def accuracy(self) -> Tuple[float, float]:
        """
        calculate the accuracy of the labeled data for sentiws and textblob
        :return: a tuple of the accuracy for sentiws and textblob
        """
        # calculate accuracy for sentiws
        accuracy_sentiws = metrics.accuracy_score(self.dataframe["labeled_sentiment"], self.dataframe["sentiment"])

        # calculate accuracy for textblob
        accuracy_textblob = metrics.accuracy_score(
            self.dataframe["labeled_sentiment"], self.dataframe["sentiment_textblob"]
        )

        # print the results
        print("==================== Accuracy ====================\n")
        print("SentiWS: " + str(accuracy_sentiws))
        print("Textblob: " + str(accuracy_textblob))
        return accuracy_sentiws, accuracy_textblob

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
