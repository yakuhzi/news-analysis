import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from utils.writer import Writer


class TfidfSentiment:
    def __init__(self, df_paragraphs):
        self.df_paragraphs = df_paragraphs

    def add_sentiment(self, overwrite: bool = False) -> None:
        # Sentiment already calculated
        if "sentiment_score" in self.df_paragraphs and not overwrite:
            return

        # Get text from dataframe
        self.df_paragraphs["text"] = self.df_paragraphs["text"].apply(
            lambda row: [self._remove_umlauts(word) for word in row]
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
        self.df_paragraphs["sentiment_sentiws"] = self.df_paragraphs["sentiment_sentiws"].apply(
            lambda row: [0 if x is None else x for x in row]
        )

        # Calculate sentiment from dot product of polarity and tfidf
        self.df_paragraphs["sentiment_score"] = self.df_paragraphs.apply(
            lambda row: np.dot(row["sentiment_sentiws"], row["tfidf"]), axis=1
        )

        # Save paragraphs to disk
        Writer.write_articles(self.df_paragraphs, "paragraphs")

    def map_sentiment(self, overwrite: bool = False) -> None:
        # Sentiment already mapped
        if "sentiment" in self.df_paragraphs and not overwrite:
            return

        # Map sentiment values to "Positive", "Negative" or "Neutral"
        self.df_paragraphs["sentiment"] = self.df_paragraphs["sentiment_score"].apply(
            lambda score: self._map_sentiment(score)
        )

        # Save paragraphs to disk
        Writer.write_articles(self.df_paragraphs, "paragraphs")

    def _map_sentiment(self, score: str) -> str:
        score = float(score)

        if score > 0.001:
            return "Positive"
        elif score < -0.001:
            return "Negative"
        else:
            return "Neutral"

    def _remove_umlauts(self, string):
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
