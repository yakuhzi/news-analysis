import itertools
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import nltk
import numpy as np
import pandas as pd
import spacy
from spacy.lang.de.stop_words import STOP_WORDS
from spacy_sentiws import spaCySentiWS


class PreprocessArticles:
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

    def lowercase_article(self, articles):
        articles["text"] = articles["text"].str.lower()

    def extract_paragraphs(self, articles, df_preprocessed_articles):
        print(articles.head())
        texts = articles["text"]
        paragraph_list = list(map(lambda text: text.replace("\n\n", "\n").split("\n"), texts))
        flat_list = [(index, item) for index, sublist in enumerate(paragraph_list) for item in sublist]
        list_indices, list_paragraph_texts = zip(*flat_list)
        df_preprocessed_articles["article_index"] = list_indices
        df_preprocessed_articles["text"] = list_paragraph_texts
        print(df_preprocessed_articles.head(100))

    def remove_special_characters(self, df_preprocessed_articles):
        df_preprocessed_articles["text"] = df_preprocessed_articles["text"].str.replace(r"[^A-Za-z0-9äöüÄÖÜß\- ]", " ")
        # df_preprocessed_articles["paragraphs"] = df_preprocessed_articles["paragraphs"].apply(
        #     lambda row: [re.sub(r"[^A-Za-z0-9äöüÄÖÜß\- ]", "", paragraph) for paragraph in row]
        # )

    def remove_stopwords(self, df_preprocessed_articles):
        stopwords = spacy.lang.de.stop_words.STOP_WORDS
        df_preprocessed_articles["text"] = df_preprocessed_articles["text"].apply(
            lambda words: " ".join(word for word in words.split() if word not in stopwords)
        )
        # df_preprocessed_articles["paragraphs"] = df_preprocessed_articles["paragraphs"].apply(
        #     lambda row: [" ".join(word for word in paragraph.split(" ") if word not in stopwords) for paragraph in row]
        # )

    def tokenization(self, df_preprocessed_articles):
        df_preprocessed_articles["text"] = df_preprocessed_articles["text"].apply(lambda x: self.nlp(x))

    def pos_tagging(self, df_preprocessed_articles):
        df_preprocessed_articles["pos_tags"] = df_preprocessed_articles["text"].apply(
            lambda row: [(word, word.tag_) for word in row]
        )

    def sentiws(self, df_preprocessed_articles):
        df_preprocessed_articles["sentiws"] = df_preprocessed_articles["text"].apply(
            lambda row: [(word, word._.sentiws) for word in row]
        )

    def lemmatizing(self, df_preprocessed_articles):
        df_preprocessed_articles["lemma"] = df_preprocessed_articles["text"].apply(
            lambda row: [word.lemma_ for word in row]
        )

    def find_parties(self, row: pd.Series) -> pd.Series:
        organizations = [x.lower() for x in row["organizations_ner"]]
        # print("-------------------")
        # print(row)
        # print("-------------------")

        party_list = []
        for organization in organizations:
            for key, value in self.parties.items():
                if organization in value and key not in party_list:
                    party_list.append(key)
        row["parties"] = party_list
        return row

    def concat_lemma(self, df_preprocessed_articles):
        df_preprocessed_articles["lemma"] = df_preprocessed_articles["lemma"].apply(lambda row: [" ".join(row)])

    def tag_dataframe(self, row: pd.Series) -> pd.Series:
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
        person_list = PreprocessArticles.filter_out_synonyms(person_list, 3)
        organization_list = PreprocessArticles.filter_out_synonyms(organization_list, 1)
        return person_list, organization_list

    @staticmethod
    def filter_out_synonyms(ner_list: List[str], biggest_allowed_distance: int) -> List[str]:
        new_ner_list = list(dict.fromkeys(ner_list))
        # print(ner_list)
        # print(new_ner_list)
        # print(len(ner_list))
        # print(len(new_ner_list))
        return new_ner_list

    def preprocessing(self, articles: pd.DataFrame, split_paragraphs=True):
        if split_paragraphs:
            df_preprocessed_articles = pd.DataFrame({"text": []})
            self.extract_paragraphs(articles, df_preprocessed_articles)
        else:
            df_preprocessed_articles = articles.copy()
            df_preprocessed_articles["article_index"] = df_preprocessed_articles.index

        # remove special characters (regex)
        self.remove_special_characters(df_preprocessed_articles)

        # de_core_news_lg had the best score for entity recognition and syntax accuracy in german according to spacy.
        # for more information, see https://spacy.io/models/de#de_core_news_lg
        self.nlp = spacy.load("de_core_news_lg", disable=["parser"])
        sentiws = spaCySentiWS(sentiws_path="src/data/sentiws/")
        print("Number of articles: {}".format(len(df_preprocessed_articles)))
        # NER Tagging for persons and organizations
        df_preprocessed_articles = df_preprocessed_articles.apply(self.tag_dataframe, axis=1)

        df_preprocessed_articles = df_preprocessed_articles.apply(self.find_parties, axis=1)
        df_preprocessed_articles = df_preprocessed_articles.loc[
            np.array(list(map(len, df_preprocessed_articles.parties.values))) > 0
        ]
        print("Number of articles 2: {}".format(len(df_preprocessed_articles)))
        # lowercase everything
        self.lowercase_article(df_preprocessed_articles)

        # stop word removal (after POS? -> filter unwanted POS)
        # evtl verbessern (opel fährt auf transporter auf --> opel fährt transporter)

        self.remove_stopwords(df_preprocessed_articles)

        # tokenization
        self.tokenization(df_preprocessed_articles)

        self.nlp.add_pipe(sentiws)
        self.sentiws(df_preprocessed_articles)
        # POS tagging (before stemming? Could be used to count positive or negative adjectives etc.
        self.pos_tagging(df_preprocessed_articles)

        # stemming or lemmatization
        self.lemmatizing(df_preprocessed_articles)
        self.concat_lemma(df_preprocessed_articles)

        return df_preprocessed_articles
