import pandas as pd


class Preprocess_Articles:
    def preprocessing(self, articles: pd.DataFrame):
        preprocessed_articles = articles.copy()
        preprocessed_articles = preprocessed_articles[:10]
        print(preprocessed_articles)
        """remove special characters (regex)"""
        """stop word removal (after POS? -> filter unwanted POS)"""
        """ tokenization """
        """ POS tagging (before stemming? Could be used to count positive or negative adjectives etc. """
        """ stemming or lemmatization"""
        """ BoW or TF-IDF? """
