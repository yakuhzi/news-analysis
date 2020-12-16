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

    def write_articles(self, article: pd.DataFrame, filename: str):
        """
        writes the dataframes of the preprocessed articles to a json file
        """
        path = "data/" + filename + ".json"
        article.to_json(path, orient="columns", default_handler=str)
