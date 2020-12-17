from typing import List, Tuple

import pandas as pd
import spacy
from spacy.lang.de.stop_words import STOP_WORDS


def lowercase_article(articles):
    articles["text"] = articles["text"].str.lower()


def remove_special_characters(df_preprocessed_articles):
    df_preprocessed_articles["text"] = df_preprocessed_articles["text"].str.replace(
        r"[^a-z0-9äöü ]", ""
    )


def remove_stopwords(df_preprocessed_articles):
    stopwords = spacy.lang.de.stop_words.STOP_WORDS
    df_preprocessed_articles["text"] = df_preprocessed_articles["text"].apply(
        lambda words: " ".join(word for word in words.split() if word not in stopwords)
    )


def tokenization(df_preprocessed_articles, nlp):
    df_preprocessed_articles["text"] = df_preprocessed_articles["text"].apply(
        lambda x: nlp(x)
    )


def pos_tagging(df_preprocessed_articles):
    df_preprocessed_articles["pos_tags"] = df_preprocessed_articles["text"].apply(
        lambda row: [(word, word.tag_) for word in row]
    )


def lemmatizing(df_preprocessed_articles):
    df_preprocessed_articles["lemma"] = df_preprocessed_articles["text"].apply(
        lambda row: [word.lemma_ for word in row]
    )


def tag_dataframe(row: pd.Series) -> pd.Series:
    """
    Function to apply on Pandas data frame that it is tagged

    Arguments:
    - row: the current row of the data frame to be tagged

    Return:
    - row: Pandas series with the tagged text in colums 'persons' and 'rows'
    """
    persons, organizations = tag(row.text)
    row["persons"] = persons
    row["organizations"] = organizations
    return row


def tag(content: str) -> Tuple[List[str], List[str]]:
    """
    Searches for Names and Organizations in texts in order to identify relevant articles with political parties

    Arguments:
    - content: The text to search for the Named Entities

    Return:
    - person_list: List of recognized persons in the text.
    - organization_list: List of organizations in the text.
    """
    #  de_core_news_lg had the best score for entity recognition in german according to spacy.
    #  for more information, see https://spacy.io/models/de#de_core_news_lg
    nlp = spacy.load("de_core_news_lg", disable=["tagger", "parser"])
    doc = nlp(content)
    #  search for persons and apply filter that only persons remain in list
    filtered_persons = filter(lambda entity: entity.label_ == "PER", doc.ents)
    person_list = list(map(lambda entity: entity.text, filtered_persons))
    #  search for organizations and apply filter that only persons remain in list
    filtered_organizations = filter(lambda entity: entity.label_ == "ORG", doc.ents)
    organization_list = list(map(lambda entity: entity.text, filtered_organizations))
    return person_list, organization_list


class PreprocessArticles:
    def preprocessing(self, articles: pd.DataFrame):
        df_preprocessed_articles = articles.copy()
        df_preprocessed_articles = df_preprocessed_articles[:10]

        """ lowercase everything"""
        lowercase_article(df_preprocessed_articles)

        """remove special characters (regex)"""
        remove_special_characters(df_preprocessed_articles)

        """stop word removal (after POS? -> filter unwanted POS)"""
        # evtl verbessern (opel fährt auf transporter auf --> opel fährt transporter)
        nlp = spacy.load("de_core_news_lg", disable=["parser", "ner"])
        remove_stopwords(df_preprocessed_articles)

        """ tokenization """
        tokenization(df_preprocessed_articles, nlp)

        """ POS tagging (before stemming? Could be used to count positive or negative adjectives etc. """
        pos_tagging(df_preprocessed_articles)

        """ stemming or lemmatization"""
        lemmatizing(df_preprocessed_articles)

        return df_preprocessed_articles
