from typing import List, Tuple

import pandas as pd
import spacy


class NERTagger:
    """
    Class that tags dataframes using spacy.

    Attributes:
    - count: temporary for debugging to check performance
    """

    def __init__(self):
        self.count = 0

    def tag_dataframe(self, row: pd.Series) -> pd.Series:
        """
        Function to apply on Pandas data frame that it is tagged

        Arguments:
        - row: the current row of the data frame to be tagged

        Return:
        - row: Pandas series with the tagged text in colums 'persons' and 'rows'
        """
        persons, organizations = self.tag(row.text)
        row["persons"] = persons
        row["organizations"] = organizations
        return row

    def tag(self, content: str) -> Tuple[List[str], List[str]]:
        """
        Searches for Names and Organizations in texts in order to identify relevant articles with political parties

        Arguments:
        - content: The text to search for the Named Entities

        Return:
        - person_list: List of recognized persons in the text.
        - organization_list: List of organizations in the text.
        """
        self.count += 1
        print(self.count)
        #  de_core_news_lg had the best score for entity recognition in german according to spacy.
        #  for more information, see https://spacy.io/models/de#de_core_news_lg
        nlp = spacy.load("de_core_news_lg", disable=["tagger", "parser"])
        doc = nlp(content)
        #  search for persons and apply filter that only persons remain in list
        filtered_persons = filter(lambda entity: entity.label_ == "PER", doc.ents)
        person_list = list(map(lambda entity: entity.text, filtered_persons))
        #  search for organizations and apply filter that only persons remain in list
        filtered_organizations = filter(lambda entity: entity.label_ == "ORG", doc.ents)
        organization_list = list(
            map(lambda entity: entity.text, filtered_organizations)
        )
        return person_list, organization_list
