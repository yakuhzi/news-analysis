import json

import pandas as pd


class Writer:
    """
    Class that writes the preprocessed news articles to a json file.

    Attributes:
    - df_bild_articles: A pandas dataframe of all BILD news articles.
    - df_tagesschau_articles: A pandas dataframe of all Tagesschau news articles.
    - df_taz_articles: A pandas dataframe of all TAZ news articles.
    """

    @staticmethod
    def write_articles(df: pd.DataFrame, filename: str):
        """
        Helper function to store Pandas dataframe into json file

        Arguments:
        - df: the Pandas data frame which should be stored in json
        - filename: the path where the dataframe should be stored
        """
        path = "src/data/" + filename + ".json"
        with open(path, "w", encoding="utf-8") as file:
            df.to_json(file, force_ascii=False, orient="records", default_handler=str)
