import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta
from pandas import DataFrame


def get_term_count_nouns(df_interval_paragraphs, term):
    """
    count nouns which are equal to the term provided in the given dataframe
    :param df_interval_paragraphs: dataframe, containing only the entries in a given timeframe
    :param term: term, that is searched for in the dataframe
    :return: number of found nouns, divided by the lenght of the given paragraph
    """
    count = df_interval_paragraphs["nouns"].apply(lambda row: row.count(term)).sum()
    if len(df_interval_paragraphs.index) == 0:
        return count
    return count / len(df_interval_paragraphs.index)


def get_term_count_overall(df_interval_paragraphs, term):
    """
    count words which are equal to the term provided in the given dataframe
    :param df_interval_paragraphs: dataframe, containing only the entries in a given timeframe
    :param term: term, that is searched for in the dataframe
    :return: number of found words, divided by the lenght of the given paragraph
    """
    count = df_interval_paragraphs["text"].apply(lambda row: row.count(term)).sum()
    if len(df_interval_paragraphs.index) == 0:
        return count
    return count / len(df_interval_paragraphs.index)


class TimeCourse:
    def __init__(self):
        self.df_paragraphs = None

    def set_paragraph(self, df_paragraph):
        self.df_paragraphs = df_paragraph

    def get_time_course(
        self,
        party_list: list,
        media_list: list,
        df_top_terms: DataFrame,
        initial_start_date: datetime,
        initial_end_date: datetime,
        df_paragraph: DataFrame,
    ):
        """
        calculate the data, which is later needed to show a diagram. Therefore the number of occurances of several top_terms
        have to be determined within a specific dataframe. The time difference is always one month
        :param party_list: list of parties
        :param media_list: list of media outlets
        :param df_top_terms: top terms for each party
        :param initial_start_date: start date where time should should be calculated
        :param initial_end_date: end date where time should should be calculated
        :param df_paragraph: dataframe containing the text
        :return: dataframes containing the data to plot the time course
        """
        self.df_paragraphs = df_paragraph
        df_image = pd.DataFrame(columns=["party", "media", "term", "weight"])
        for party in party_list:
            for media in media_list:
                # use only top 3 words for each party
                df_party_term = df_top_terms[df_top_terms["party"] == party]
                for term in df_party_term["term"]:
                    # split time window into smaller chunks
                    # calculate months between dates
                    start_date = initial_start_date
                    next_end_date = initial_start_date + relativedelta(months=+1)
                    weight_list = []
                    dates = []
                    while next_end_date < initial_end_date:
                        # get weight for each month
                        df_interval_paragraphs = self.configure_dataframe_for_time_course(
                            start_date, next_end_date, media, df_paragraph
                        )
                        weight = get_term_count_nouns(df_interval_paragraphs, term)
                        dates.append(start_date)
                        weight_list.append(weight)
                        start_date = next_end_date
                        next_end_date = start_date + relativedelta(months=+1)
                    df_image = df_image.append(
                        {"party": party, "media": media, "term": term, "weight": weight_list, "dates": dates},
                        ignore_index=True,
                    )
        return df_image

    def get_time_course_custom_word(
        self,
        filter_list: list,
        word: str,
        filter_criteria_word: str,
        initial_start_date: datetime,
        initial_end_date: datetime,
        df_paragraphs: DataFrame,
    ):
        """
        calculate the data, which is later needed to show a diagram. Therefore the number of occurances of a word
        has to be determined within a specific dataframe. The time difference is always one month
        :param filter_list: list containing either media outlet or party
        :param word: word, that the occurence should be calculated on
        :param filter_criteria_word: specify whether it is filtered by media outlet or by party
        :param initial_start_date: start date where time should should be calculated
        :param initial_end_date: end date where time should should be calculated
        :param df_paragraphs: dataframe containing the text
        :return: dataframes containing the data to plot the time course
        """
        df_image = pd.DataFrame(columns=["filter_criteria", "word", "weight", "dates"])
        for filter_criteria in filter_list:
            if filter_criteria_word == "media":
                df_term = df_paragraphs[df_paragraphs["media"] == filter_criteria]
            else:
                df_term = df_paragraphs[df_paragraphs.apply(lambda row: filter_criteria in row["parties"], axis=1)]
            start_date = initial_start_date
            next_end_date = initial_start_date + relativedelta(months=+1)
            weight_list = []
            dates = []
            while next_end_date < initial_end_date:
                df_interval_paragraphs = self.configure_dataframe_for_custom_time_course(
                    start_date, next_end_date, filter_criteria_word, filter_criteria, df_term
                )
                weight = get_term_count_overall(df_interval_paragraphs, word)
                weight_list.append(weight)
                dates.append(start_date)
                start_date = next_end_date
                next_end_date = start_date + relativedelta(months=+1)
            df_image = df_image.append(
                {"filter_criteria": filter_criteria, "word": word, "weight": weight_list, "dates": dates},
                ignore_index=True,
            )
        return df_image

    def configure_dataframe_for_custom_time_course(
        self, start_date, end_date, filter_criteria_word, filter_criteria, df
    ):
        """
        configure the dataframe for each month that needs to be calculated
        :param start_date: start date of interval timeframe
        :param end_date: end date of interval timeframe
        :param filter_criteria_word: specify whether it is filtered by media outlet or by party
        :param filter_criteria:
        :param df: dataframe where operations shouldbe performed on
        :return: dataframe containing only the determined time intervall
        """
        df_paragraphs_time_interval = df[df["date"].notna()]
        if filter_criteria_word == "media":
            df_paragraphs_time_interval = df_paragraphs_time_interval[
                df_paragraphs_time_interval["media"] == filter_criteria
            ]
        else:
            df_paragraphs_time_interval = df_paragraphs_time_interval[
                df_paragraphs_time_interval.apply(lambda row: filter_criteria in row["parties"], axis=1)
            ]
        if start_date and end_date:
            start = start_date.strftime("%Y-%m-%d")
            end = end_date.strftime("%Y-%m-%d")
            df_paragraphs_time_interval = df_paragraphs_time_interval[
                (df_paragraphs_time_interval["date"] > start) & (df_paragraphs_time_interval["date"] < end)
            ]
        return df_paragraphs_time_interval

    def configure_dataframe_for_time_course(self, start_date, end_date, media, df):
        """
        configure the dataframe for each month that needs to be calculated
        :param start_date: start date of interval timeframe
        :param end_date: end date of interval timeframe
        :param media: determines
        :param df: dataframe, where the operation is performed on
        :return: dataframe containing only the determined time intervall
        """
        df_paragraphs_time_interval = df[df["date"].notna()]
        df_paragraphs_time_interval = df_paragraphs_time_interval[df_paragraphs_time_interval["media"] == media]
        if start_date and end_date:
            start = start_date.strftime("%Y-%m-%d")
            end = end_date.strftime("%Y-%m-%d")
            df_paragraphs_time_interval = df_paragraphs_time_interval[
                (df_paragraphs_time_interval["date"] > start) & (df_paragraphs_time_interval["date"] < end)
            ]
        return df_paragraphs_time_interval
