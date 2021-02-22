import re
import warnings
from typing import List

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pandas import DataFrame
from pandas.core.common import SettingWithCopyWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class KeywordExtraction:
    """
    class to get most important keywords/ topics in newspapers for different parties
    """

    def __init__(self, df_paragraphs: DataFrame):
        self.df_paragraphs = df_paragraphs

    def set_data(self, df_paragraphs: DataFrame) -> None:
        """
        sets the dataframe where to determine the topics from
        :param df_paragraphs: the dataframe to analyze
        """
        self.df_paragraphs = df_paragraphs

    def set_active_media(self, media_list: list) -> None:
        """
        filters the media where to extract the keywords from (TAZ, Tagesschau or BILD)
        :param media_list: media to analyze
        """
        self.df_paragraphs = self.df_paragraphs[self.df_paragraphs["media"].isin(media_list)]

    def get_term_weight_tuples(self, parties: list = None, topn: int = 3) -> DataFrame:
        """
        get most important keywords by TF-IDF weights and count their appearance
        :param parties: the parties of which the topic should be analyzed
        :param topn: how many top words should be chosen for each party
        :return: a data frame with the most important term-weight-tuples for the respective parties
        """
        # Get nouns from dataframe
        nouns = self.df_paragraphs["nouns"].apply(lambda row: " ".join(row))

        # Vectorize nouns
        vectorizer = CountVectorizer(min_df=5, token_pattern=r"(?u)\b[a-zA-Z0-9_\-][a-zA-Z0-9_\-]+\b")
        count_vectorized = vectorizer.fit_transform(nouns)

        # Apply tf-idf to count_vectorized
        transformer = TfidfTransformer(smooth_idf=True, use_idf=True)

        # Generate tf-idf for the given document
        transformer.fit(count_vectorized)

        if parties is None:
            parties = ["CDU", "CSU", "SPD", "FDP", "AfD", "Grüne", "Linke"]

        terms = []
        # get top words by TF-IDF weight
        for party in parties:
            terms += self._get_top_words(topn, party, vectorizer, transformer)

        terms = list(set(terms))
        tuples = []
        # count appearance of term for party to determine weight for bipartite graph
        for party in parties:
            for term in terms:
                term_count = self.get_term_count(self.df_paragraphs, party, term)
                tuples.append((party, term, term_count))

        return DataFrame(tuples, columns=["party", "term", "weight"])

    def get_top_terms_for_party(self, parties: list = None) -> DataFrame:
        # Get nouns from dataframe
        nouns = self.df_paragraphs["nouns"].apply(lambda row: " ".join(row))

        # Vectorize character_words
        vectorizer = CountVectorizer(min_df=5, token_pattern=r"(?u)\b[a-zA-Z0-9_\-][a-zA-Z0-9_\-]+\b")
        count_vectorized = vectorizer.fit_transform(nouns)

        # Apply tf-idf to count_vectorized
        transformer = TfidfTransformer(smooth_idf=True, use_idf=True)

        # Generate tf-idf for the given document
        transformer.fit(count_vectorized)

        if parties is None:
            parties = ["CDU", "CSU", "SPD", "FDP", "AfD", "Grüne", "Linke"]

        tuples = []

        for party in parties:
            terms = self._get_top_words(party, vectorizer, transformer)
            terms = list(set(terms))
            for term in terms:
                tuples.append((party, term))

        return DataFrame(tuples, columns=["party", "term"])

    def _get_top_words(
        self, topn: int, party: str, vectorizer: CountVectorizer, transformer: TfidfTransformer
    ) -> List[str]:
        """
        get the most important words for a party by TF-IDF score
        :param party: party to get keywords from
        :param vectorizer: the TF-IDF Count vectorizer to use for calculating TF-IDF scores
        :param transformer: the TF-IDF transformer to use for calculating TF-IDF scores
        :return: the top 3 terms for the party as list
        """
        party_paragraphs = self._get_party_paragraphs(self.df_paragraphs, party)
        party_paragraphs["nouns"] = self._remove_blacklist_words(party_paragraphs)

        nouns = party_paragraphs["nouns"].apply(lambda row: " ".join(row))

        # transform nouns
        tf_idf_vector = transformer.transform(vectorizer.transform(nouns))
        weights = np.asarray(tf_idf_vector.mean(axis=0)).ravel().tolist()

        # create dataframe with weights and sort in descending order
        df_weights = DataFrame({"term": vectorizer.get_feature_names(), "TF_IDF": weights})
        df_weights = df_weights.sort_values("TF_IDF", ascending=False).reset_index(drop=False)

        # transform dataframe to list and return top n elements
        top_terms = df_weights["term"].tolist()[:topn]
        print("Top {} words of {}: {}".format(topn, party, top_terms))
        return top_terms

    def get_term_count(self, df: DataFrame, party: str, term: str) -> int:
        """
        counts the appearance of a term in a data frame
        :param df: the data frame where the appearance should be counted
        :param party: the party where the term frequency should be determined
        :param term: the term which should be counted
        :return: number of appearances of the specified term as integer
        """
        paragraphs = self._get_party_paragraphs(df, party)
        return paragraphs["nouns"].apply(lambda row: row.count(term)).sum()

    def get_graph(self, df_term_weights: DataFrame) -> Figure:
        """
        creates a bipartite graph fom the term-weight-tuples which were previously calculated
        :param df_term_weights: the term-weight-tuples dataframe
        :return: the bipartite graph as figure
        """
        graph = nx.Graph()
        graph.add_nodes_from(df_term_weights["party"], bipartite=0)
        graph.add_nodes_from(df_term_weights["term"], bipartite=1)
        graph.add_weighted_edges_from(
            [(row["party"], row["term"], row["weight"]) for idx, row in df_term_weights.iterrows()], weight="weight"
        )

        df_term_weights["normalized_weight"] = df_term_weights["weight"].apply(
            lambda row: row / df_term_weights["weight"].max() * 4
        )

        plt.close()
        fig = plt.figure(1, figsize=(10, 9))

        nx.draw(
            graph,
            pos=nx.drawing.layout.bipartite_layout(graph, df_term_weights["party"]),
            width=df_term_weights["normalized_weight"].tolist(),
            with_labels=True,
        )

        return fig

    def _get_party_paragraphs(self, dataframe: DataFrame, party: str) -> DataFrame:
        """
        filter paragraphs to only contain rows with a certain party
        :param dataframe: the dataframe to filter
        :param party: the party to keep in dataframe
        :return: the filtered dataframe containing only rows about the specified party
        """
        return dataframe[dataframe.apply(lambda row: len(row["parties"]) == 1 and party in row["parties"], axis=1)]

    def _remove_blacklist_words(self, dataframe: DataFrame) -> DataFrame:
        """
        remove words which are not meaningful as keywords/ topics e.g. ministry positions that are held by a certain
        party, weekdays etc.
        :param dataframe: the dataframe to filter the nouns
        :return: the filtered data frame
        """
        return dataframe["nouns"].apply(self._filter_words)

    def _filter_words(self, words: List[str]) -> List[str]:
        """
        filter words (remove words contained in blacklist
        :param words: words to filter
        :return: filtered word list
        """
        blacklist_words = [
            "minister",
            "kanzler",
            "bürgermeister",
            "montag",
            "dienstag",
            "mittwoch",
            "donnerstag",
            "freitag",
            "samstag",
            "sonntag",
            "cdu",
            "csu",
            "spd",
            "grüne",
            "linke",
            "afd",
            "fdp",
            "partei",
            "sprecher",
            "bundesregierung",
            "union",
            "jahr",
            "live-ticker",
            "info",
            "ob",
            "liberale",
            "tag",
            "uhr",
            "spalte",
            "artikel",
        ]

        blacklist = re.compile("|".join([re.escape(word) for word in blacklist_words]))
        return [word for word in words if not blacklist.search(word)]
