import pandas as pd
import spacy
from spacy.lang.de.stop_words import STOP_WORDS


class Preprocess_Articles:
    def preprocessing(self, articles: pd.DataFrame):
        df_preprocessed_articles = articles.copy()
        df_preprocessed_articles = df_preprocessed_articles[:10]

        """ lowercase everything"""
        df_preprocessed_articles["text"] = df_preprocessed_articles["text"].str.lower()
        print(df_preprocessed_articles["text"])

        """remove special characters (regex)"""
        df_preprocessed_articles["text"] = df_preprocessed_articles["text"].str.replace(
            r"[^a-z0-9äöü ]", ""
        )

        """stop word removal (after POS? -> filter unwanted POS)"""
        # evtl verbessern (opel fährt auf transporter auf --> opel fährt transporter)
        nlp = spacy.load("de_core_news_lg", disable=["parser", "ner"])
        stopwords = spacy.lang.de.stop_words.STOP_WORDS
        df_preprocessed_articles["text"] = df_preprocessed_articles["text"].apply(
            lambda words: " ".join(
                word for word in words.split() if word not in stopwords
            )
        )

        """ tokenization """
        df_preprocessed_articles["text"] = df_preprocessed_articles["text"].apply(
            lambda x: nlp(x)
        )

        """ POS tagging (before stemming? Could be used to count positive or negative adjectives etc. """
        pos_tags = []
        for item in df_preprocessed_articles["text"]:
            pos_tag = [(i, i.tag_) for i in item]
            pos_tags.append(pos_tag)
        df_preprocessed_articles["pos_tags"] = pos_tags
        """ stemming or lemmatization"""
        """ BoW or TF-IDF? """

        print(df_preprocessed_articles["pos_tags"])
