from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series
from sklearn import metrics

from tfidf_sentiment import TfidfSentiment
from utils.reader import Reader
from utils.writer import Writer


class Comparison:
    def __init__(self, path: str):
        self.dataframe = Reader.read_json_to_df_default(path)
        self.tfidf_sentiment = TfidfSentiment(self.dataframe)

    def train_threshold(self) -> float:
        """
        Train the threshold with labeled data.

        :return The best threshold.
        """
        threshold: float = 0
        best_threshold: float = 0
        best_score: float = 0

        thresholds: List[float] = []
        f1_scores: List[Tuple[float, float, float, float]] = []

        self.tfidf_sentiment.get_context_polarity(8)
        self.tfidf_sentiment.calculate_sentiment_score(overwrite=True)

        # Iterate over different thresholds, increase with every loop
        while threshold <= 0.005:
            self.tfidf_sentiment.map_sentiment(threshold=threshold, overwrite=True)

            # Optimize over the sum of all f1 scores for SentiWs
            f1_sentiws, _, _ = self.f1_score(training=True)
            f1_sum = f1_sentiws[0] + f1_sentiws[1] + f1_sentiws[2]

            thresholds.append(threshold)
            f1_scores.append((f1_sum, f1_sentiws[0], f1_sentiws[1], f1_sentiws[2]))

            # Replace best threshold if current one is better
            if f1_sum > best_score:
                best_score = f1_sum
                best_threshold = threshold

            threshold += 0.0000001

        # Visualize the training
        self.visualize_threshold(thresholds, f1_scores, best_threshold, 0.005)

        # Adjust the sentiment with best threshold
        self.tfidf_sentiment.map_sentiment(threshold=best_threshold, overwrite=True)
        Writer.write_dataframe(self.dataframe, "labeled_paragraphs")
        return best_threshold

    def train_context_thresholds(self) -> Tuple[float, float]:
        """
        Train the context threshold (SentiWs with context polarity) with labeled data.

        :return The best thresholds for window size and score.
        """
        window_threshold: int = 0
        best_window_threshold: int = 0
        best_score_threshold: float = 0
        best_score: float = 0

        thresholds: List[float] = []
        f1_scores: List[Tuple] = []

        # Iterate over different window thresholds, increase with every loop
        while window_threshold <= 35:
            self.tfidf_sentiment.get_context_polarity(window_threshold)
            self.tfidf_sentiment.calculate_sentiment_score(overwrite=True)

            score_threshold: float = 0

            # Save best temp scores for visualization
            best_temp_score_f1_sum: float = 0
            best_temp_score_f1_scores: Tuple = ()

            # Iterate over different score thresholds, increase with every loop
            while score_threshold < 0.001:
                self.tfidf_sentiment.map_sentiment(overwrite=True, threshold=score_threshold)
                self.dataframe = self.tfidf_sentiment.df_paragraphs

                # Optimize over the sum of all f1 scores for context sentiment
                _, _, f1_context = self.f1_score(training=True)
                f1_sum = f1_context[0] + f1_context[1] + f1_context[2]

                # Replace best temp thresholds for visualization if current ones are better
                if f1_sum > best_temp_score_f1_sum:
                    best_temp_score_f1_sum = f1_sum
                    best_temp_score_f1_scores = (f1_sum, f1_context[0], f1_context[1], f1_context[2])

                # Replace best thresholds if current ones are better
                if f1_sum > best_score:
                    best_score = f1_sum
                    best_window_threshold = window_threshold
                    best_score_threshold = score_threshold

                score_threshold += 0.00001

            thresholds.append(window_threshold)
            f1_scores.append(best_temp_score_f1_scores)

            window_threshold += 1

        # Visualize the training
        self.visualize_threshold(thresholds, f1_scores, best_window_threshold, 35)

        # Adjust the sentiment with best thresholds
        self.tfidf_sentiment.get_context_polarity(best_window_threshold)
        self.tfidf_sentiment.calculate_sentiment_score(overwrite=True)
        self.tfidf_sentiment.map_sentiment(overwrite=True, threshold=best_score_threshold)
        Writer.write_dataframe(self.dataframe, "labeled_paragraphs")
        return best_window_threshold, best_score_threshold

    def visualize_threshold(
        self,
        threshold: List[float],
        f1_scores: List[Tuple[float, float, float, float]],
        best_threshold: float,
        max_x: float,
    ) -> None:
        """
        Visualize the results of training the threshold.

        :param threshold: All thresholds used for training as list.
        :param f1_scores: The f1 scores for (total, positive, negative, neutral) labeled data.
        :param best_threshold: The best threshold from training.
        :param max_x: The biggest x value of the data.
        """
        plt.axis((0, max_x, 0, 2))

        # Plot f1 scores as line graphs
        plt.plot(threshold, [scores[0] for scores in f1_scores], color="black", label="f1 sum")
        plt.plot(threshold, [scores[1] for scores in f1_scores], color="green", label="f1 positive")
        plt.plot(threshold, [scores[2] for scores in f1_scores], color="orange", label="f1 negative")
        plt.plot(threshold, [scores[3] for scores in f1_scores], color="blue", label="f1 neutral")

        # Plot the best found threshold as vertical line
        plt.axvline(best_threshold, color="red", label="best threshold")
        plt.legend()
        plt.show()

    def precision(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the precision of the labeled data for SentiWS, TextBlob and the context sentiment.

        :return: Tuple of the precision scores for positive, negative and neutral labeled data for SentiWs, TextBlob
        and the context sentiment.
        """
        labels = ["Positive", "Negative", "Neutral"]

        # calculate precision for SentiWs
        precision_sentiws = metrics.precision_score(
            self.dataframe["labeled_sentiment"], self.dataframe["sentiment"], labels=labels, average=None
        )

        # calculate precision for textblob
        precision_textblob = metrics.precision_score(
            self.dataframe["labeled_sentiment"], self.dataframe["sentiment_textblob"], labels=labels, average=None
        )

        precision_context = metrics.precision_score(
            self.dataframe["labeled_sentiment"], self.dataframe["sentiment_context"], labels=labels, average=None
        )

        # Print the result
        self._output_metric(
            metric="Precision",
            columns=["precision_sentiws", "precision_textblob", "precision_context"],
            results=[precision_sentiws, precision_textblob, precision_context],
        )

        return precision_sentiws, precision_textblob, precision_context

    def recall(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the recall of the labeled data for SentiWS, TextBlob and the context sentiment.

        :return: Tuple of the recall scores for positive, negative and neutral labeled data for SentiWs, TextBlob
        and the context sentiment.
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

        recall_context = metrics.recall_score(
            self.dataframe["labeled_sentiment"], self.dataframe["sentiment"], labels=labels, average=None
        )

        # Print the result
        self._output_metric(
            metric="Recall",
            columns=["recall_sentiws", "recall_textblob", "recall_context"],
            results=[recall_sentiws, recall_textblob, recall_context],
        )

        return recall_sentiws, recall_textblob, recall_context

    def f1_score(self, training: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the f1 score of the labeled data for SentiWS, TextBlob and the context sentiment.

        :param training: True if threshold is trained (do not print score)
        :return: Tuple of the f1 scores for positive, negative and neutral labeled data for SentiWs, TextBlob
        and the context sentiment.
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

        # calculate context f1 score
        f1_context = metrics.f1_score(
            self.dataframe["labeled_sentiment"], self.dataframe["sentiment_context"], labels=labels, average=None
        )

        # Print the result (not if threshold is trained)
        if not training:
            self._output_metric(
                metric="F1-score",
                columns=["f1_sentiws", "f1_textblob", "f1_context"],
                results=[f1_sentiws, f1_textblob, f1_context],
            )

        return f1_sentiws, f1_textblob, f1_context

    def accuracy(self) -> Tuple[float, float, float]:
        """
        Calculate the accuracy of the labeled data for SentiWS, TextBlob and the context sentiment.

        :return: Tuple of the accuracy for SentiWs, TextBlob and the context sentiment.
        """
        # calculate accuracy for sentiws
        accuracy_sentiws = metrics.accuracy_score(self.dataframe["labeled_sentiment"], self.dataframe["sentiment"])

        # calculate accuracy for textblob
        accuracy_textblob = metrics.accuracy_score(
            self.dataframe["labeled_sentiment"], self.dataframe["sentiment_textblob"]
        )

        accuracy_context = metrics.accuracy_score(
            self.dataframe["labeled_sentiment"], self.dataframe["sentiment_context"]
        )

        # Print the results
        print("==================== Accuracy ====================\n")
        print("SentiWS: " + str(accuracy_sentiws))
        print("Textblob: " + str(accuracy_textblob))
        print("Context Sentiment: " + str(accuracy_context))
        return accuracy_sentiws, accuracy_textblob, accuracy_context

    def _output_metric(
        self,
        metric: str,
        columns: List[str],
        results: List[np.ndarray],
    ) -> None:
        """
        Prints the result of metrics calculation as pandas data frame.

        :param metric: The metric to print.
        :param columns: The column names for the SentiWs, TextBlob and context sentiment results.
        :param results: The resulting metric for SentiWs, TextBlob and context sentiment.
        """
        data = {
            columns[0]: [results[0][0], results[0][1], results[0][2], results[0][0] + results[0][1] + results[0][2]],
            columns[1]: [results[1][0], results[1][1], results[1][2], results[1][0] + results[1][1] + results[1][2]],
            columns[2]: [results[2][0], results[2][1], results[2][2], results[2][0] + results[2][1] + results[2][2]],
        }

        dataframe = pd.DataFrame(data=data, index=["Positive", "Negative", "Neutral", "Sum"])
        print("==================== " + metric + " ====================\n")
        print(dataframe.to_markdown() + "\n\n")

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
