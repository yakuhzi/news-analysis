import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from utils.writer import Writer


class TfidfSentiment:
    def __init__(self, df_paragraphs):
        self.df_paragraphs = df_paragraphs

    def add_sentiment(self) -> DataFrame:
        # Get text from dataframe
        self.df_paragraphs["text"] = self.df_paragraphs["text"].apply(
            lambda row: [self._remove_umlaut(word) for word in row]
        )

        text = self.df_paragraphs["text"].apply(lambda row: " ".join(row))

        # Vectorize character_words
        vectorizer = CountVectorizer(tokenizer=lambda y: y.split())
        count_vectorized = vectorizer.fit_transform(text)

        # Apply tf-idf to count_vectorized
        transformer = TfidfTransformer(smooth_idf=True, use_idf=True)

        # Generate tf-idf for the given document
        tf_idf_vector = transformer.fit_transform(count_vectorized)

        # Get tf-idf weights
        weights = np.asarray(tf_idf_vector.mean(axis=0)).ravel().tolist()

        # Get term-tfidf dictionary
        dict_weights = dict(zip(vectorizer.get_feature_names(), weights))

        # Save tfidf score for each word
        self.df_paragraphs["tfidf"] = self.df_paragraphs["text"].apply(
            lambda row: [dict_weights[word] for word in row if word != " "]
        )

        # Replace nan polarity values with 0
        self.df_paragraphs["polarity"] = self.df_paragraphs["polarity"].apply(
            lambda row: [0 if x is None else x for x in row]
        )

        # Calculate sentiment from dot product of polarity and tfidf
        self.df_paragraphs["sentiment"] = self.df_paragraphs.apply(
            lambda row: np.dot(row["polarity"], row["tfidf"]), axis=1
        )

        # Map sentiment values to "Positive", "Negative" or "Neutral"
        self.df_paragraphs["sentiment"] = self.df_paragraphs["sentiment"].apply(lambda row: self._map_sentiment(row))

        # Save paragraphs to disk
        Writer.write_articles(self.df_paragraphs, "paragraphs")
        return self.df_paragraphs

    def _map_sentiment(self, row):
        if row > 0:
            return "Positive"
        elif row < 0:
            return "Negative"
        else:
            return "Neutral"

    def _remove_umlaut(self, string):
        """
        Removes umlauts from strings and replaces them with the letter+e convention
        :param string: string to remove umlauts from
        :return: unumlauted string
        """
        string = string.encode()

        string = string.replace("ü".encode(), b"ue")
        string = string.replace("Ü".encode(), b"Ue")
        string = string.replace("ä".encode(), b"ae")
        string = string.replace("Ä".encode(), b"Ae")
        string = string.replace("ö".encode(), b"oe")
        string = string.replace("Ö".encode(), b"Oe")
        string = string.replace("ß".encode(), b"ss")

        string = string.decode("utf-8")
        return string
