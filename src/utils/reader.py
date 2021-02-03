import json

import pandas as pd
from pandas import DataFrame


class Reader:
    """
    Class that reads the news articles from the json files.
    """

    @staticmethod
    def read_articles(number_of_samples: int = None) -> DataFrame:
        """
        Reads the news article for every news agency and returns them.
        """
        df_tagesschau_articles = Reader.read("src/data/tagesschau.json")
        df_tagesschau_articles["media"] = "Tagesschau"

        df_taz_articles = Reader.read("src/data/taz.json")
        df_taz_articles["media"] = "TAZ"

        df_bild_articles = Reader.read("src/data/bild.json")
        df_bild_articles["media"] = "Bild"

        df_articles = df_tagesschau_articles.append(df_taz_articles).append(df_bild_articles)

        if number_of_samples is not None:
            df_articles = df_articles.sample(number_of_samples).reset_index(drop=True)

        print("Number of articles: {}".format(len(df_articles)))
        return df_articles

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

    @staticmethod
    def read(path: str) -> pd.DataFrame:
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
