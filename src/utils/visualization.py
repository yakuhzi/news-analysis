import math
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from pandas import DataFrame

from utils.statistics import Statistics


class Visualization:
    """
    Helper class for sentiment visualization
    """

    color_dict = {
        "Total": "#1f77b4",
        "Tagesschau": "#2ca02c",
        "TAZ": "#ff7f0e",
        "Bild": "#1f77b4",
        "CDU": "black",
        "CSU": "darkgrey",
        "SPD": "orangered",
        "Grüne": "green",
        "FDP": "yellow",
        "AfD": "blue",
        "Linke": "red",
    }

    @staticmethod
    def get_basic_statistic_bar_plot(dataframe: DataFrame, parties: List[str], media: List[str]) -> Figure:
        """
        Get the basic statistic (document distribution) figures.

        :param dataframe: Dataframe to extract the statistics from.
        :param parties: Parties to consider for the statistics.
        :param media: Media to consider for the statistics.
        :return: Figures containing the bar charts for each party.
        """
        x, y = Statistics.get_basic_statistics(dataframe, media, parties)
        return Visualization._get_document_distribution_figure(x, y)

    @staticmethod
    def get_media_statistics_bar_plots(dataframe: DataFrame, parties: List[str], media: List[str]) -> List[Figure]:
        """
        Get the basic statistic (document distribution) figures of each media.

        :param dataframe: Dataframe to extract the statistics from.
        :param parties: Parties to consider for the statistics.
        :param media: Media to consider for the statistics.
        :return: Figures containing the bar charts for each party.
        """
        figures = []

        for outlet in media:
            x, y = Statistics.get_media_statistics(dataframe, outlet, parties)
            fig = Visualization._get_document_distribution_figure(x, y, outlet)
            figures.append(fig)

        return figures

    @staticmethod
    def get_party_statistics_bar_plots(dataframe: DataFrame, parties: List[str], media: List[str]) -> List[Figure]:
        """
        Get the basic statistic (document distribution) figures of each party.

        :param dataframe: Dataframe to extract the statistics from.
        :param parties: Parties to consider for the statistics.
        :param media: Media to consider for the statistics.
        :return: Figures containing the bar charts for each party.
        """
        figures = []

        for party in parties:
            x, y = Statistics.get_party_statistics(dataframe, party, media)
            fig = Visualization._get_document_distribution_figure(x, y, party)
            figures.append(fig)

        return figures

    @staticmethod
    def _get_document_distribution_figure(x: List[str], y: List[int], party_or_media: Optional[str] = None) -> Figure:
        """
        Get a bar chart figure for the document distribution.

        :param x: X values of the plot.
        :param y: Y values of the plot.
        :param party_or_media: Name of the party or media used for the plot title.
        :return: Bar chart figure representing the document distribution.
        """
        color_dict = Visualization.color_dict.copy()
        color_dict["Tagesschau"] = "grey"
        color_dict["TAZ"] = "grey"
        color_dict["Bild"] = "grey"

        colors = []

        # Map colors of each bar
        for item in x:
            colors.append(color_dict[item])

        # Construct bar plot
        fig, ax = plt.subplots(1, 1)
        bar_plot = ax.bar(x, y, color=colors)
        ax.set_frame_on(False)
        ax.tick_params("x", labelrotation=90, length=0)
        ax.axes.yaxis.set_ticks([])
        ax.set_ylabel("Number of documents")

        # Add values (number of documents) on top of each bar
        for idx, rect in enumerate(bar_plot):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.0, 1.05 * height, y[idx], ha="center", va="bottom", rotation=0)

        # Set figure title
        if party_or_media is None:
            fig.suptitle("Document Distribution")
        else:
            fig.suptitle("Document Distribution for {}".format(party_or_media))

        fig.tight_layout()
        return fig

    @staticmethod
    def get_sentiment_pie_charts(
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
        statistics = Statistics.get_sentiment_statistics(df_paragraphs, by_party, parties, media)

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
    def get_time_course_plots(df_time_course: DataFrame):
        figures = []
        colors = {"Tagesschau": "#2ca02c", "TAZ": "#ff7f0e", "Bild": "#1f77b4"}

        # Define terms for grouping
        different_terms = df_time_course.term.unique()
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
                    axs.set_xticks(dates_array)
                    axs.set_xlabel("Months")
                    axs.set_ylabel("Frequency of Usage")
                    axs.legend(loc="best", title="Outlet", frameon=False)
            plt.setp(axs.xaxis.get_majorticklabels(), rotation=25)
            figures.append(fig)
        return figures

    @staticmethod
    def get_time_course_plots_custom_word(df_time_course: DataFrame):
        figures = []

        # Define terms for grouping
        different_filter = df_time_course.filter_criteria.unique()
        word = df_time_course["word"][0]

        fig, axs = plt.subplots(1, 1)
        fig.suptitle("Usage of term {0}".format(word))
        # lines to draw
        for filter_criteria in different_filter:
            df_plot = df_time_course[df_time_course["filter_criteria"] == filter_criteria]
            if not df_plot.empty:
                weights = df_plot["weight"]
                weights_array = np.asarray(weights)[0]
                # get right color for media
                line_color = Visualization.color_dict[filter_criteria]
                dates = df_plot["dates"]
                dates_array = np.asarray(dates)[0]
                axs.plot(dates_array, weights_array, color=line_color, label=filter_criteria)
                axs.set_xticks(dates_array)
                axs.set_xlabel("Months")
                axs.set_ylabel("Frequency of Usage")
                axs.legend(loc="best", title="Outlet", frameon=False)
        plt.setp(axs.xaxis.get_majorticklabels(), rotation=25)
        figures.append(fig)
        return figures
