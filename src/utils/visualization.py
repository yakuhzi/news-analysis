import math
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from pandas import DataFrame


class Visualization:
    @staticmethod
    def show_pie_charts(df_paragraphs: DataFrame, by_party: bool = True):
        statistics = Visualization.get_statistics(df_paragraphs, by_party)
        labels = ["Positive", "Negative", "Neutral"]
        colors = ["#2ca02c", "#ff7f0e", "#1f77b4"]

        for key_1, value_1 in statistics.items():
            lines = []

            if by_party:
                fig, axs = plt.subplots(1, len(value_1))
                fig.suptitle("Sentiment towards {}".format(key_1))
            else:
                fig, axs = plt.subplots(2, math.ceil(len(value_1) / 2))
                fig.suptitle("Sentiment of {} towards parties".format(key_1))
                axs = [item for sublist in axs for item in sublist]

            for (key_2, value_2), ax in zip(value_1.items(), axs):
                ax.axis("equal")
                ax.set_title(key_2)
                line = ax.pie(value_2, colors=colors, counterclock=False, autopct="%1.1f%%", shadow=True, startangle=90)
                lines.append(line)

            fig.legend(labels=labels, loc="lower right", borderaxespad=0.1, title="Sentiment")

        plt.show()

    @staticmethod
    def get_statistics(df_paragraphs: DataFrame, by_party: bool) -> Dict[str, Dict[str, Tuple[int, int, int]]]:
        statistics: Dict[str, Dict[str, Tuple[int, int, int]]] = {}
        parties = ["CDU", "CSU", "SPD", "AfD", "Grüne", "Linke"]
        media = ["Tagesschau", "TAZ", "Bild"]

        for item_1 in parties if by_party else media:
            party_statistics: Dict[str, Tuple[int, int, int]] = {}

            for item_2 in media if by_party else parties:
                df_paragraphs_filtered = df_paragraphs[
                    (df_paragraphs["media"] == (item_2 if by_party else item_1))
                    & (df_paragraphs["parties"].apply(lambda row: (item_1 if by_party else item_2) in row))
                ]

                total_number = len(df_paragraphs_filtered)

                if total_number == 0:
                    party_statistics[item_2] = (0, 0, 0)
                    continue

                positive = len(df_paragraphs_filtered.loc[(df_paragraphs_filtered["sentiment"] == "Positive")])
                negative = len(df_paragraphs_filtered.loc[(df_paragraphs_filtered["sentiment"] == "Negative")])
                neutral = len(df_paragraphs_filtered.loc[(df_paragraphs_filtered["sentiment"] == "Neutral")])

                party_statistics[item_2] = (positive, negative, neutral)

            statistics[item_1] = party_statistics

        print(statistics)
        return statistics
