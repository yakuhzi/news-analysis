import pandas as pd


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

        """ tokenization """
        """ POS tagging (before stemming? Could be used to count positive or negative adjectives etc. """
        """ stemming or lemmatization"""
        """ BoW or TF-IDF? """
