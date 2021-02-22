import pandas as pd
from pandas import DataFrame
import datetime
from dateutil.relativedelta import relativedelta

from keyword_extraction import KeywordExtraction

class TimeCourse:

    def __init__(self):
        self.df_paragraphs = None

    def get_time_course(self, party_list: list, media_list: list, df_top_terms: DataFrame,
                        keyword_extraction: KeywordExtraction, initial_start_date: datetime, initial_end_date: datetime,
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
                            start_date, next_end_date, media
                        )
                        weight = keyword_extraction.get_term_count(df_interval_paragraphs, True, party, term)
                        dates.append(start_date)
                        weight_list.append(weight)
                        start_date = next_end_date
                        next_end_date = start_date + relativedelta(months=+1)
                    df_image = df_image.append(
                        {"party": party, "media": media, "term": term, "weight": weight_list, "dates": dates},
                        ignore_index=True,
                    )
        return df_image

    def configure_dataframe_for_time_course(self, start_date, end_date, media):
        df_paragraphs_time_interval = self.df_paragraphs[self.df_paragraphs["date"].notna()]
        df_paragraphs_time_interval = df_paragraphs_time_interval[df_paragraphs_time_interval["media"] == media]
        if start_date and end_date:
            start = start_date.strftime("%Y-%m-%d")
            end = end_date.strftime("%Y-%m-%d")
            df_paragraphs_time_interval = df_paragraphs_time_interval[
                (df_paragraphs_time_interval["date"] > start) & (df_paragraphs_time_interval["date"] < end)
                ]
        return df_paragraphs_time_interval
