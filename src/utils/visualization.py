import datetime
import math
from typing import Dict, List, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
import matplotlib.ticker as mticker
from pandas import DataFrame


class Visualization:
    """
    Helper class for sentiment visualization
    """

    @staticmethod
    def get_pie_charts(
        df_paragraphs: DataFrame, by_party: bool = True, parties: List[str] = None, media: List[str] = None
    ) -> List[Figure]:
        """
        Get figures of pie charts for the sentiment either grouped by party or by media outlet.

        :param df_paragraphs: the dataframe of the paragraphs
        :param by_party: If True, group data by party, otherwise group by media
        :param parties: List of parties to consider. Defaults to all parties.
        :param media: List of media outlets to consider. Defaults to all media outlets.

        :return: List of figures containing the pie charts
        """

        # Get sentiment statistics
        statistics = Visualization.get_statistics(df_paragraphs, by_party, parties, media)

        # Define label and colors for pie charts
        labels = ["Positive", "Negative", "Neutral"]
        colors = ["#2ca02c", "#ff7f0e", "#1f77b4"]

        figures = []

        for key_1, value_1 in statistics.items():

            if by_party:
                # Group pie charts by party
                fig, axs = plt.subplots(1, len(value_1))
                fig.suptitle("Sentiment towards {}".format(key_1))

                if len(value_1) == 1:
                    axs = [axs]

                figures.append(fig)
            else:
                # Group pie charts by media
                rows = 2 if len(value_1) > 2 else 1
                columns = math.ceil(len(value_1) / 2) if len(value_1) > 2 else len(value_1)
                fig, axs = plt.subplots(rows, columns)
                fig.suptitle("Sentiment of {} towards parties".format(key_1))

                if len(value_1) % 2 == 1 and len(value_1) > 1:
                    fig.delaxes(axs.flatten()[-1])
                if len(value_1) > 2:
                    axs = [item for sublist in axs for item in sublist]
                if len(value_1) == 1:
                    axs = [axs]

                figures.append(fig)

            # Create a pie chart plot for each item in the group
            for (key_2, value_2), ax in zip(value_1.items(), axs):
                ax.axis("equal")
                ax.set_title(key_2)
                ax.pie(value_2, colors=colors, counterclock=False, autopct="%1.1f%%", shadow=True, startangle=90)

            # Add legend to plot
            fig.legend(labels=labels, loc="lower right", borderaxespad=0.1, title="Sentiment")

        return figures

    @staticmethod
    def get_plots(df_time_course: DataFrame):
        figures = []
        colors = {"Tagesschau": "#2ca02c", "TAZ": "#ff7f0e", "Bild": "#1f77b4"}

        # Define terms for grouping
        different_terms = df_time_course.term.unique()
        different_parties = df_time_course.party.unique()
        different_media = df_time_course.media.unique()

        for term in different_terms:
            fig, axs = plt.subplots(1, 1)
            fig.suptitle("Usage of term {0}".format(term))
            # plots to draw
            df_step1 = df_time_course[df_time_course["term"] == term]
            # lines to draw
            for media in different_media:
                df_plot = df_step1[df_step1["media"] == media]
                if not df_plot.empty:
                    weights = df_plot["weight"]
                    weights_array = np.asarray(weights)[0]
                    # get right color for media
                    line_color = colors[media]
                    dates = df_plot["dates"]
                    dates_array = np.asarray(dates)[0]
                    axs.plot(dates_array, weights_array, color=line_color, label=media)
                    axs.set_xlabel("Months")
                    axs.set_ylabel("Frequency of Usage")
                    axs.legend(loc="best", title="Outlet", frameon=False)
            figures.append(fig)
        return figures

    @staticmethod
    def get_plots_custom_word(df_time_course: DataFrame):
        figures = []
        colors = {"Tagesschau": "#2ca02c", "TAZ": "#ff7f0e", "Bild": "#1f77b4"}

        # Define terms for grouping
        different_media = df_time_course.media.unique()
        word = df_time_course["word"][0]

        fig, axs = plt.subplots(1, 1)
        fig.suptitle("Usage of term {0}".format(word))
        # lines to draw
        for media in different_media:
            df_plot = df_time_course[df_time_course["media"] == media]
            if not df_plot.empty:
                weights = df_plot["weight"]
                weights_array = np.asarray(weights)[0]
                # get right color for media
                line_color = colors[media]
                dates = df_plot["dates"]
                dates_array = np.asarray(dates)[0]
                axs.plot(dates_array, weights_array, color=line_color, label=media)
                axs.set_xlabel("Months")
                axs.set_ylabel("Frequency of Usage")
                axs.legend(loc="best", title="Outlet", frameon=False)
        figures.append(fig)
        return figures

    @staticmethod
    def get_statistics(
        df_paragraphs: DataFrame, by_party: bool, parties: List[str], media: List[str]
    ) -> Dict[str, Dict[str, Tuple[int, int, int]]]:
        """
        Get statistics for the sentiment either grouped by party or by media outlet.

        :param df_paragraphs: the dataframe of the paragraphs
        :param by_party: If True, group data by party, otherwise group by media
        :param parties: List of parties to consider. Defaults to all parties.
        :param media: List of media outlets to consider. Defaults to all media outlets.

        :return: Dictionary containing the statistics
        """
        statistics: Dict[str, Dict[str, Tuple[int, int, int]]] = {}

        # Define parties for grouping
        if parties is None:
            parties = ["CDU", "CSU", "SPD", "AfD", "Grüne", "Linke"]

        # Define media for grouping
        if media is None:
            media = ["Tagesschau", "TAZ", "Bild"]

        # Iterate over parties or media
        for item_1 in parties if by_party else media:
            party_statistics: Dict[str, Tuple[int, int, int]] = {}

            # Iterate over media or parties
            for item_2 in media if by_party else parties:
                # Get paragraphs filtered by party and media
                df_paragraphs_filtered = df_paragraphs[
                    (df_paragraphs["media"] == (item_2 if by_party else item_1))
                    & (df_paragraphs["parties"].apply(lambda row: (item_1 if by_party else item_2) in row))
                ]

                # Get number of paragraphs filtered by party and media
                total_number = len(df_paragraphs_filtered)

                if total_number == 0:
                    party_statistics[item_2] = (0, 0, 0)
                    continue

                # Get number of positive, negative and neutral sentences
                positive = len(df_paragraphs_filtered.loc[(df_paragraphs_filtered["sentiment"] == "Positive")])
                negative = len(df_paragraphs_filtered.loc[(df_paragraphs_filtered["sentiment"] == "Negative")])
                neutral = len(df_paragraphs_filtered.loc[(df_paragraphs_filtered["sentiment"] == "Neutral")])

                party_statistics[item_2] = (positive, negative, neutral)

            statistics[item_1] = party_statistics

        print(statistics)
        return statistics
