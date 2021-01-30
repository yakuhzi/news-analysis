import json
from typing import Optional

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

    def read_articles(self, number: int = None) -> None:
        """
        Reads the news article for every news agency and stores it in the corresponding instance variables.
        """

        self.df_bild_articles = self._read("src/data/bild.json")
        self.df_tagesschau_articles = self._read("src/data/tagesschau.json")
        self.df_taz_articles = self._read("src/data/taz.json")

        if number is not None:
            self.df_bild_articles = self.df_bild_articles.head(number)
            self.df_tagesschau_articles = self.df_tagesschau_articles.head(number)
            self.df_taz_articles = self.df_taz_articles.head(number)

        print("Number of Bild articles: {}".format(len(self.df_bild_articles.index)))
        print("Number of Tagesschau articles: {}".format(len(self.df_tagesschau_articles.index)))
        print("Number of TAZ articles: {}".format(len(self.df_taz_articles.index)))

    @staticmethod
    def read_json_to_df_default(path: str) -> pd.DataFrame:
        """
        Read a json into a Pandas dataframe without any modifications on types.

        Arguments:
        - path: the path of the json file
        - set_article_index: if True, the original article index is set as index in the data frame

        Return:
        - Pandas data frame build from the json file
        """
        with open(path, encoding="utf8") as json_file:
            json_dict = json.load(json_file)
            df = pd.DataFrame(json_dict)

            if "article_index" in json_dict:
                df.set_index("article_index", inplace=True, drop=True)

            return df

    def _read(self, path: str) -> pd.DataFrame:
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
