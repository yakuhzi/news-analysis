from typing import List, Tuple

import numpy as np
import pandas as pd
import spacy
from spacy.lang.de.stop_words import STOP_WORDS
from utils.document_type import DocumentType


class Preprocessing:
    def __init__(self):
        self.nlp = None
        self.parties = {
            "CDU": ["cdu", "union"],
            "CSU": ["csu"],
            "SPD": ["spd", "sozialdemokraten"],
            "Grüne": [
                "grüne",
                "grünen",
                "die grüne",
                "die grünen",
                "den grünen",
                "bündnis90 die grünen",
            ],
            "FDP": [
                "fdp",
                "liberalen",
                "freien demokrate",
                "freie demokratische partei",
            ],
            "AfD": ["afd", "alternative für deutschland"],
            "Linke": ["linke", "die linke", "den linken"],
        }

    def lowercase(self, series: pd.Series) -> pd.Series:
        return series.str.lower()

    def remove_special_characters(self, series: pd.Series) -> pd.Series:
        return series.str.replace(r"[^A-Za-z0-9äöüÄÖÜß\- ]", " ")

    def remove_stopwords(self, series: pd.Series) -> pd.Series:
        stopwords = spacy.lang.de.stop_words.STOP_WORDS
        return series.apply(lambda words: " ".join(word for word in words.split() if word not in stopwords))

    def tokenization(self, series: pd.Series) -> pd.Series:
        return series.apply(lambda x: self.nlp(x))

    def pos_tagging(self, series: pd.Series) -> pd.Series:
        return series.apply(lambda row: [(word, word.tag_) for word in row])

    def lemmatizing(self, series: pd.Series) -> pd.Series:
        return series.apply(lambda row: [word.lemma_ for word in row])

    def concat_lemma(self, series: pd.Series) -> pd.Series:
        return series.apply(lambda row: [" ".join(row)])

    def tag_dataframe(self, row):
        """
        Function to apply on Pandas data frame that it is tagged

        Arguments:
        - row: the current row of the data frame to be tagged

        Return:
        - row: Pandas series with the tagged text in colums 'persons' and 'rows'
        """
        persons, organizations = self.tag(row.text)
        row["persons_ner"] = persons
        row["organizations_ner"] = organizations
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
        if self.nlp is None:
            self.nlp = spacy.load("de_core_news_lg", disable=["parser"])
        doc = self.nlp(content)
        #  search for persons and apply filter that only persons remain in list
        filtered_persons = filter(lambda entity: entity.label_ == "PER", doc.ents)
        person_list = list(map(lambda entity: entity.text, filtered_persons))
        #  search for organizations and apply filter that only persons remain in list
        filtered_organizations = filter(lambda entity: entity.label_ == "ORG", doc.ents)
        organization_list = list(map(lambda entity: entity.text, filtered_organizations))
        person_list = Preprocessing.filter_out_synonyms(person_list, 3)
        organization_list = Preprocessing.filter_out_synonyms(organization_list, 1)
        return person_list, organization_list

    def extract_paragraphs(self, articles, df_preprocessed_articles):
        print(articles.head())
        texts = articles["text"]
        paragraph_list = list(map(lambda text: text.replace("\n\n", "\n").split("\n"), texts))
        flat_list = [(index, item) for index, sublist in enumerate(paragraph_list) for item in sublist]
        list_indices, list_paragraph_texts = zip(*flat_list)
        df_preprocessed_articles["article_index"] = list_indices
        df_preprocessed_articles["text"] = list_paragraph_texts
        print(df_preprocessed_articles.head(100))

    def find_parties(self, row):
        organizations = [x.lower() for x in row["organizations_ner"]]

        party_list = []
        for organization in organizations:
            for key, value in self.parties.items():
                if organization in value and key not in party_list:
                    party_list.append(key)

        row["parties"] = party_list
        return row

    def filter_parties(self, dataframe: pd.DataFrame):
        df_preprocessed = dataframe.apply(self.find_parties, axis=1)
        return df_preprocessed.loc[np.array(list(map(len, df_preprocessed.parties.values))) > 0]

    @staticmethod
    def filter_out_synonyms(ner_list: List[str], biggest_allowed_distance: int) -> List[str]:
        new_ner_list = list(dict.fromkeys(ner_list))
        # print(ner_list)
        # print(new_ner_list)
        # print(len(ner_list))
        # print(len(new_ner_list))
        return new_ner_list

    def preprocess_titles(self, articles: pd.DataFrame) -> pd.DataFrame:
        self.nlp = spacy.load("de_core_news_lg", disable=["parser"])
        df_preprocessed = articles[["title"]]

        df_preprocessed["title"] = self.remove_special_characters(df_preprocessed["title"])
        df_preprocessed["title"] = self.lowercase(df_preprocessed["title"])
        df_preprocessed["title"] = self.remove_stopwords(df_preprocessed["title"])
        df_preprocessed["title"] = self.tokenization(df_preprocessed["title"])
        df_preprocessed["pos_tags"] = self.pos_tagging(df_preprocessed["title"])

        return df_preprocessed

    def preprocessing(self, articles: pd.DataFrame, document_type: DocumentType) -> pd.DataFrame:
        # de_core_news_lg had the best score for entity recognition and syntax accuracy in german according to spacy.
        # for more information, see https://spacy.io/models/de#de_core_news_lg
        self.nlp = spacy.load("de_core_news_lg", disable=["parser"])

        if document_type.value == DocumentType.TITLE.value:
            return self.preprocess_titles(articles)
        elif document_type.value == DocumentType.PARAGRAPH.value:
            df_preprocessed = pd.DataFrame({"text": []})
            self.extract_paragraphs(articles, df_preprocessed)
        else:
            df_preprocessed = articles.copy()
            df_preprocessed["article_index"] = df_preprocessed.index

        # remove special characters (regex)
        df_preprocessed["text"] = self.remove_special_characters(df_preprocessed["text"])

        print("Number of articles: {}".format(len(df_preprocessed)))

        # NER Tagging for persons and organizations
        df_preprocessed = df_preprocessed.apply(self.tag_dataframe, axis=1)

        # filter articles with no parties
        df_preprocessed = self.filter_parties(df_preprocessed)

        print("Number of articles 2: {}".format(len(df_preprocessed)))

        # lowercase everything
        df_preprocessed["text"] = self.lowercase(df_preprocessed["text"])

        # stop word removal (after POS? -> filter unwanted POS)
        # evtl verbessern (opel fährt auf transporter auf --> opel fährt transporter)
        df_preprocessed["text"] = self.remove_stopwords(df_preprocessed["text"])

        # tokenization
        df_preprocessed["text"] = self.tokenization(df_preprocessed["text"])

        # POS tagging (before stemming? Could be used to count positive or negative adjectives etc.
        df_preprocessed["pos_tags"] = self.pos_tagging(df_preprocessed["text"])

        # lemmatization
        df_preprocessed["lemma"] = self.lemmatizing(df_preprocessed["text"])
        df_preprocessed["lemma"] = self.concat_lemma(df_preprocessed["lemma"])

        return df_preprocessed
