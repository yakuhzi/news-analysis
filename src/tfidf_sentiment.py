import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class TfidfSentiment:
    """
    Class that calculates the sentiment of the paragraphs.
    """

    def __init__(self, df_paragraphs):
        self.df_paragraphs = df_paragraphs

    def calculate_sentiment_score(self, overwrite: bool = False) -> None:
        """
        Calculate sentiment score for each row in the dataframe and add it into the column "sentiment_score".

        :param overwrite: If True, overwrites the current sentiment.
        """

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
        self.df_paragraphs["polarity"] = self.df_paragraphs["polarity"].apply(
            lambda row: [0 if x is None else x for x in row]
        )

        # Calculate sentiment score from dot product of polarity and tfidf
        self.df_paragraphs["sentiment_score"] = self.df_paragraphs.apply(
            lambda row: np.dot(row["polarity"], row["tfidf"]), axis=1
        )

    def map_sentiment(self, threshold: float = 8.27e-05, overwrite: bool = False) -> None:
        """
        Maps the polarity of SentiWs and TextBlob to "Positive", "Negative" or "Neutral" for all paragraphs.
        :param threshold: the threshold to decide when to map positive/negative or neutral
        :param overwrite: If True, overwrites the current sentiment.
        """

        # Sentiment already mapped
        if "sentiment" not in self.df_paragraphs or overwrite:
            # Map sentiment score to "Positive", "Negative" or "Neutral"
            self.df_paragraphs["sentiment"] = self.df_paragraphs["sentiment_score"].apply(
                lambda score: self._map_sentiment(score, threshold)
            )

        if "sentiment_textblob" not in self.df_paragraphs or overwrite:
            # Map sentiment score to "Positive", "Negative" or "Neutral"
            self.df_paragraphs["sentiment_textblob"] = self.df_paragraphs["polarity_textblob"].apply(
                lambda score: self._map_sentiment(score, threshold)
            )

    def _map_sentiment(self, score: str, threshold: float = 8.27e-05) -> str:
        """
        Helper function that maps the sentiment_score to "Positive", "Negative" or "Neutral".
        :param score: The calculated sentiment_score of a paragraph
        :param threshold: the threshold to decide when to map positive/negative or neutral.
        :return: "Positive", "Negative" or "Neutral" dependent of the score input.
        """
        score = float(score)

        if score > threshold:
            return "Positive"
        elif score < -threshold:
            return "Negative"
        else:
            return "Neutral"

    def _remove_umlauts(self, string: str) -> str:
        """
        Removes umlauts from strings and replaces them with the letter+e convention.

        :param string: String to remove umlauts from.
        :return: Unumlauted string.
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
