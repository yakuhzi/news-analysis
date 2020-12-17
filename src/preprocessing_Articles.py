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
