import json
from typing import List, Optional


class Reader:
    """
    Class that reads the news articles from the json files.

    Attributes:
    - bild_articles: A list of all BILD news articles.
    - tagesschau_articles: A list of all Tagesschau news articles.
    - taz_articles: A list of all TAZ news articles.
    """

    def __init__(self):
        self.bild_articles: Optional[List[dict]] = None
        self.tagesschau_articles: Optional[List[dict]] = None
        self.taz_articles: Optional[List[dict]] = None

    def read_articles(self) -> None:
        """
        Reads the news article for every news agency and stores it in the corresponding instance variables.
        """

        self.bild_articles = self.__read("src/data/bild.json")
        self.tagesschau_articles = self.__read("src/data/tagesschau.json")
        self.taz_articles = self.__read("src/data/taz.json")

    def __read(self, path: str) -> List[dict]:
        """
        Helper function to read a json from a file.

        Arguments:
        - path: Path to json file.

        Return:
        - articles: List of JSON articles parsed from the input file.
        """

        with open(path, encoding="utf8") as json_file:
            return json.load(json_file)["articles"]
