import pandas as pd
import spacy
from spacy.lang.de.stop_words import STOP_WORDS


def lowercase_article(articles):
    articles["text"] = articles["text"].str.lower()
    return articles


def remove_special_characters(df_preprocessed_articles):
    df_preprocessed_articles["text"] = df_preprocessed_articles["text"].str.replace(
        r"[^a-z0-9äöü ]", ""
    )
    return df_preprocessed_articles


def remove_stopwords(df_preprocessed_articles):
    stopwords = spacy.lang.de.stop_words.STOP_WORDS
    df_preprocessed_articles["text"] = df_preprocessed_articles["text"].apply(
        lambda words: " ".join(word for word in words.split() if word not in stopwords)
    )
    return df_preprocessed_articles


def tokenization(df_preprocessed_articles, nlp):
    df_preprocessed_articles["text"] = df_preprocessed_articles["text"].apply(
        lambda x: nlp(x)
    )
    return df_preprocessed_articles


def pos_tagging(df_preprocessed_articles):
    pos_tags = []
    for item in df_preprocessed_articles["text"]:
        pos_tag = [(i, i.tag_) for i in item]
        pos_tags.append(pos_tag)
    df_preprocessed_articles["pos_tags"] = pos_tags
    return df_preprocessed_articles


def lemmatizing(df_preprocessed_articles):
    lemmatized = []
    for item in df_preprocessed_articles["text"]:
        lemma = [(i, i.lemma_) for i in item]
        lemmatized.append(lemma)
    df_preprocessed_articles["lemma"] = lemmatized
    return df_preprocessed_articles


class Preprocess_Articles:
    def preprocessing(self, articles: pd.DataFrame):
        df_preprocessed_articles = articles.copy()
        df_preprocessed_articles = df_preprocessed_articles[:10]

        """ lowercase everything"""
        df_preprocessed_articles = lowercase_article(df_preprocessed_articles)

        """remove special characters (regex)"""
        df_preprocessed_articles = remove_special_characters(df_preprocessed_articles)

        """stop word removal (after POS? -> filter unwanted POS)"""
        # evtl verbessern (opel fährt auf transporter auf --> opel fährt transporter)
        nlp = spacy.load("de_core_news_lg", disable=["parser", "ner"])
        df_preprocessed_articles = remove_stopwords(df_preprocessed_articles)

        """ tokenization """
        df_preprocessed_articles = tokenization(df_preprocessed_articles, nlp)

        """ POS tagging (before stemming? Could be used to count positive or negative adjectives etc. """
        df_preprocessed_articles = pos_tagging(df_preprocessed_articles)

        """ stemming or lemmatization"""
        df_preprocessed_articles = lemmatizing(df_preprocessed_articles)

        return df_preprocessed_articles
