import json
from typing import List, Optional

import pandas as pd


class Reader:
    """
    Class that reads the news articles from the json files.

    Attributes:
    - df_bild_articles: A pandas dataframe of all BILD news articles.
    - df_tagesschau_articles: A pandas dataframe of all Tagesschau news articles.
    - df_taz_articles: A pandas dataframe of all TAZ news articles.
    """

    def __init__(self):
        self.df_bild_articles: Optional[pd.DataFrame] = None
        self.df_tagesschau_articles: Optional[pd.DataFrame] = None
        self.df_taz_articles: Optional[pd.DataFrame] = None

    def read_articles(self) -> None:
        """
        Reads the news article for every news agency and stores it in the corresponding instance variables.
        """

        self.df_bild_articles = self.__read("src/data/bild.json")
        self.df_tagesschau_articles = self.__read("src/data/tagesschau.json")
        self.df_taz_articles = self.__read("src/data/taz.json")

    @staticmethod
    def __read(path: str) -> pd.DataFrame:
        """
        Helper function to read a json from a file and store it in pandas dataframe.

        Arguments:
        - path: Path to json file.

        Return:
        - articles: Panda data frame of JSON articles parsed from the input file.
        """

        with open(path, encoding="utf8") as json_file:
            json_dict = json.load(json_file)["articles"]
            return pd.DataFrame(json_dict).astype(
                {
                    "title": "string",
                    "text": "string",
                    "summary": "string",
                    "date": "string",
                    "authors": "object",
                    "references": "object",
                }
            )
