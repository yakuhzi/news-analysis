import numpy as np
from pandas import Series

from utils.reader import Reader


class Comparison:
    def __init__(self, filename: str):
        self.dataframe = Reader.read_json_to_df_default("src/output/" + filename + ".json")

    def polarity(self):
        unlabeled_series = self.dataframe["sentiment"]
        labeled_series = self.dataframe["labeled_sentiment"]

        comparison = np.where((unlabeled_series == labeled_series), True, False)
        number_of_equal = np.count_nonzero(comparison)

        comparison_neutral = np.where(
            ((unlabeled_series == labeled_series) & (unlabeled_series == "Neutral")), True, False
        )
        number_of_equal_neutral = np.count_nonzero(comparison_neutral)
        neutral = np.where(labeled_series == "Neutral", True, False)
        number_of_neutrals = np.count_nonzero(neutral)

        comparison_positive = np.where(
            ((unlabeled_series == labeled_series) & (unlabeled_series == "Positive")), True, False
        )
        number_of_equal_positives = np.count_nonzero(comparison_positive)
        positive = np.where(labeled_series == "Positive", True, False)
        number_of_positives = np.count_nonzero(positive)

        comparison_negative = np.where(
            ((unlabeled_series == labeled_series) & (unlabeled_series == "Negative")), True, False
        )
        number_of_equal_negatives = np.count_nonzero(comparison_negative)
        negative = np.where(labeled_series == "Negative", True, False)
        number_of_negatives = np.count_nonzero(negative)

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

        print("Not neutral but subjective: {}".format(subjective_series.value_counts()[1]))
        print("Not neutral but objective: {}".format(objective_series.value_counts()[1]))
