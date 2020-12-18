from typing import List, Tuple

import pandas as pd
import spacy
from spacy.lang.de.stop_words import STOP_WORDS


class PreprocessArticles:
    def __init__(self):
        self.nlp = None

    def lowercase_article(self, articles):
        articles["text"] = articles["text"].str.lower()

    def replace_new_line(self, df_preprocessed_articles):
        df_preprocessed_articles["text"] = df_preprocessed_articles["text"].str.replace(
            "\\n", " "
        )

    def remove_special_characters(self, df_preprocessed_articles):
        df_preprocessed_articles["text"] = df_preprocessed_articles["text"].str.replace(
            r"[^A-Za-z0-9äöüÄÖÜß\- ]", " "
        )

    def remove_stopwords(self, df_preprocessed_articles):
        stopwords = spacy.lang.de.stop_words.STOP_WORDS
        df_preprocessed_articles["text"] = df_preprocessed_articles["text"].apply(
            lambda words: " ".join(
                word for word in words.split() if word not in stopwords
            )
        )

    def tokenization(self, df_preprocessed_articles):
        df_preprocessed_articles["text"] = df_preprocessed_articles["text"].apply(
            lambda x: self.nlp(x)
        )

    def pos_tagging(self, df_preprocessed_articles):
        df_preprocessed_articles["pos_tags"] = df_preprocessed_articles["text"].apply(
            lambda row: [(word, word.tag_) for word in row]
        )

    def lemmatizing(self, df_preprocessed_articles):
        df_preprocessed_articles["lemma"] = df_preprocessed_articles["text"].apply(
            lambda row: [word.lemma_ for word in row]
        )

    def concat_lemma(self, df_preprocessed_articles):
        df_preprocessed_articles["lemma"] = df_preprocessed_articles["lemma"].apply(
            lambda row: [" ".join(row)]
        )

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
        organization_list = list(
            map(lambda entity: entity.text, filtered_organizations)
        )
        return person_list, organization_list

    def preprocessing(self, articles: pd.DataFrame):
        df_preprocessed_articles = articles.copy()
        df_preprocessed_articles = df_preprocessed_articles[:10]

        self.replace_new_line(df_preprocessed_articles)

        # remove special characters (regex)
        self.remove_special_characters(df_preprocessed_articles)

        # de_core_news_lg had the best score for entity recognition and syntax accuracy in german according to spacy.
        # for more information, see https://spacy.io/models/de#de_core_news_lg
        self.nlp = spacy.load("de_core_news_lg", disable=["parser"])

        # NER Tagging for persons and organizations
        df_preprocessed_articles = df_preprocessed_articles.apply(
            self.tag_dataframe, axis=1
        )

        # lowercase everything
        self.lowercase_article(df_preprocessed_articles)

        # stop word removal (after POS? -> filter unwanted POS)
        # evtl verbessern (opel fährt auf transporter auf --> opel fährt transporter)

        self.remove_stopwords(df_preprocessed_articles)

        # tokenization
        self.tokenization(df_preprocessed_articles)

        # POS tagging (before stemming? Could be used to count positive or negative adjectives etc.
        self.pos_tagging(df_preprocessed_articles)

        # stemming or lemmatization
        self.lemmatizing(df_preprocessed_articles)
        self.concat_lemma(df_preprocessed_articles)

        return df_preprocessed_articles
