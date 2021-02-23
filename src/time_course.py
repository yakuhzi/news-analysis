import pandas as pd
from pandas import DataFrame
import datetime
from dateutil.relativedelta import relativedelta

from keyword_extraction import KeywordExtraction


def get_term_count_overall(df_interval_paragraphs, term):
    return df_interval_paragraphs["nouns"].apply(lambda row: row.count(term)).sum()


class TimeCourse:

    def __init__(self):
        self.df_paragraphs = None

    def set_paragraph(self, df_paragraph):
        self.df_paragraphs = df_paragraph

    def get_time_course(self, party_list: list, media_list: list, df_top_terms: DataFrame,
                        initial_start_date: datetime, initial_end_date: datetime,
                        df_paragraph: DataFrame):
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
                        weight = get_term_count_overall(df_interval_paragraphs, term)
                        dates.append(start_date)
                        weight_list.append(weight)
                        start_date = next_end_date
                        next_end_date = start_date + relativedelta(months=+1)
                    df_image = df_image.append(
                        {"party": party, "media": media, "term": term, "weight": weight_list, "dates": dates},
                        ignore_index=True,
                    )
        return df_image

    def get_time_course_custom_word(self, media_list: list, word: str, initial_start_date: datetime, initial_end_date: datetime,
                                    df_paragraphs: DataFrame):
        df_image = pd.DataFrame(columns=["media", "word", "weight"])
        for media in media_list:
            df_term = df_paragraphs[df_paragraphs["media"] == media]
            start_date = initial_start_date
            next_end_date = initial_start_date + relativedelta(months=+1)
            weight_list = []
            dates = []
            while next_end_date < initial_end_date:
                df_interval_paragraphs = self.configure_dataframe_for_time_course(start_date, next_end_date, media, df_term)
                weight = get_term_count_overall(df_interval_paragraphs, word)
                weight_list.append(weight)
                dates.append(start_date)
                start_date = next_end_date
                next_end_date = start_date + relativedelta(months=+1)
            df_image = df_image.append(
                {"media": media, "word": word, "weight": weight_list, "dates": dates}, ignore_index=True
            )
        return df_image

    def configure_dataframe_for_time_course(self, start_date, end_date, media, df):
        df_paragraphs_time_interval = df[df["date"].notna()]
        df_paragraphs_time_interval = df_paragraphs_time_interval[df_paragraphs_time_interval["media"] == media]
        if start_date and end_date:
            start = start_date.strftime("%Y-%m-%d")
            end = end_date.strftime("%Y-%m-%d")
            df_paragraphs_time_interval = df_paragraphs_time_interval[
                (df_paragraphs_time_interval["date"] > start) & (df_paragraphs_time_interval["date"] < end)
                ]
        return df_paragraphs_time_interval
