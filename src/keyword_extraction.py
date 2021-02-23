import re
import warnings
from typing import List, Optional, Tuple

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
    Class to get most important keywords/ topics in newspapers for different parties
    """

    def __init__(self, df_paragraphs: DataFrame):
        self.df_paragraphs = df_paragraphs

    def set_data(self, df_paragraphs: DataFrame) -> None:
        """
        Sets the dataframe where to determine the topics from
        :param df_paragraphs: the dataframe to analyze
        """
        self.df_paragraphs = df_paragraphs

    def set_active_media(self, media_list: List[str]) -> None:
        """
        Filters the media where to extract the keywords from (TAZ, Tagesschau or BILD)
        :param media_list: media to analyze
        """
        self.df_paragraphs = self.df_paragraphs[self.df_paragraphs["media"].isin(media_list)]

    def get_term_weight_tuples(
        self,
        by_party: bool,
        all_terms: bool = False,
        parties: Optional[List[str]] = None,
        media: Optional[List[str]] = None,
        topn: int = 3,
    ) -> DataFrame:
        """
        Get most important keywords by TF-IDF weights and count their appearance
        :param by_party: If True, group data by party, otherwise group by media
        :param all_terms: If True, returns top terms of both media and party
        :param parties: The parties of which the topic should be analyzed
        :param media: The media of which the topic should be analyzed
        :param topn: How many top words should be chosen for each party
        :return: A data frame with the most important term-weight-tuples for the respective parties or media
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

        if media is None:
            media = ["Bild", "Tagesschau", "TAZ"]

        terms: List[str] = []

        # Get top words by TF-IDF weight
        if by_party or all_terms:
            for party in [] if parties is None else parties:
                terms += self._get_top_words(True, party, vectorizer, transformer, topn)

        if not by_party or all_terms:
            for m in [] if media is None else media:
                terms += self._get_top_words(False, m, vectorizer, transformer, topn)

        terms = list(set(terms))
        tuples: List[Tuple[str, str, int]] = []

        # Count appearance of term for party or media to determine weight for bipartite graph
        for party_or_media in parties if by_party else media:
            for term in terms:
                term_count = self.get_term_count(self.df_paragraphs, by_party, party_or_media, term)
                tuples.append((party_or_media, term, term_count))

        return DataFrame(tuples, columns=["party" if by_party else "media", "term", "weight"])

    def get_top_terms_for_party(self, parties: Optional[List[str]] = None) -> DataFrame:
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
            terms = self._get_top_words(True, party, vectorizer, transformer)
            terms = list(set(terms))
            for term in terms:
                tuples.append((party, term))

        return DataFrame(tuples, columns=["party", "term"])

    def _get_top_words(
        self,
        by_party: bool,
        party_or_media: str,
        vectorizer: CountVectorizer,
        transformer: TfidfTransformer,
        topn: int = 3,
    ) -> List[str]:
        """
        Get the most important words for a party by TF-IDF score
        :param by_party: If True, group data by party, otherwise group by media
        :param party_or_media: Party or media to get keywords from
        :param vectorizer: The TF-IDF Count vectorizer to use for calculating TF-IDF scores
        :param transformer: The TF-IDF transformer to use for calculating TF-IDF scores
        :return: The top 3 terms for the party as list
        """
        if by_party:
            paragraphs = self._get_party_paragraphs(self.df_paragraphs, party_or_media)
        else:
            paragraphs = self._get_media_paragraphs(self.df_paragraphs, party_or_media)

        # Remove blacklist words from paragraphs
        paragraphs["nouns"] = self._remove_blacklist_words(paragraphs)
        # Join nouns to single string
        nouns = paragraphs["nouns"].apply(lambda row: " ".join(row))

        # Transform nouns
        tf_idf_vector = transformer.transform(vectorizer.transform(nouns))
        weights = np.asarray(tf_idf_vector.mean(axis=0)).ravel().tolist()

        # Create dataframe with weights and sort in descending order
        df_weights = DataFrame({"term": vectorizer.get_feature_names(), "TF_IDF": weights})
        df_weights = df_weights.sort_values("TF_IDF", ascending=False).reset_index(drop=False)

        # Transform dataframe to list and return top n elements
        top_terms = df_weights["term"].tolist()[:topn]
        print("Top {} words of {}: {}".format(topn, party_or_media, top_terms))
        return top_terms

    def get_term_count(self, dataframe: DataFrame, by_party: bool, party_or_media: str, term: str) -> int:
        """
        Counts the appearance of a term in a data frame
        :param dataframe: The dataframe where the appearance should be counted
        :param by_party: If True, group data by party, otherwise group by media
        :param party_or_media: The party or media where the term frequency should be determined
        :param term: The term which should be counted
        :return: Number of appearances of the specified term as integer
        """
        if by_party:
            paragraphs = self._get_party_paragraphs(dataframe, party_or_media)
        else:
            paragraphs = self._get_media_paragraphs(dataframe, party_or_media)

        return paragraphs["nouns"].apply(lambda row: row.count(term)).sum()

    def get_graph(self, df_term_weights: DataFrame) -> Figure:
        """
        Creates a bipartite graph fom the term-weight-tuples which were previously calculated
        :param df_term_weights: The term-weight-tuples dataframe
        :return: The bipartite graph as figure
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

    def get_tripartite_graph(self, df_party_weights: DataFrame, df_media_weights: DataFrame):
        graph = nx.Graph()
        graph.add_nodes_from(df_party_weights["party"], bipartite=0)
        graph.add_nodes_from(df_party_weights["term"], bipartite=1)
        graph.add_nodes_from(df_media_weights["media"], bipartite=2)

        graph.add_weighted_edges_from(
            [(row["party"], row["term"], row["weight"]) for idx, row in df_party_weights.iterrows()],
            weight="weight",
        )

        graph.add_weighted_edges_from(
            [(row["media"], row["term"], row["weight"]) for idx, row in df_media_weights.iterrows()],
            weight="weight",
        )

        df_party_weights["normalized_weight"] = df_party_weights["weight"].apply(
            lambda row: row / df_party_weights["weight"].max() * 4
        )

        df_media_weights["normalized_weight"] = df_media_weights["weight"].apply(
            lambda row: row / df_media_weights["weight"].max() * 4
        )

        plt.close()
        fig = plt.figure(1, figsize=(10, 9))

        nx.draw(
            graph,
            pos=nx.multipartite_layout(graph, subset_key="bipartite"),
            # pos=nx.drawing.layout.bipartite_layout(graph, df_party_weights["party"]),
            width=df_party_weights["normalized_weight"].tolist(),
            with_labels=True,
        )

        return fig

    def _get_party_paragraphs(self, dataframe: DataFrame, party: str) -> DataFrame:
        """
        Filter paragraphs to only contain rows with a certain party
        :param dataframe: The dataframe to filter
        :param party: The party to keep in dataframe
        :return: The filtered dataframe containing only rows about the specified party
        """
        return dataframe[dataframe.apply(lambda row: len(row["parties"]) == 1 and party in row["parties"], axis=1)]

    def _get_media_paragraphs(self, dataframe: DataFrame, media: str) -> DataFrame:
        """
        Filter paragraphs to only contain rows with a certain party
        :param dataframe: The dataframe to filter
        :param media: The media to keep in dataframe
        :return: The filtered dataframe containing only rows about the specified party
        """
        return dataframe[dataframe.apply(lambda row: media == row["media"], axis=1)]

    def _remove_blacklist_words(self, dataframe: DataFrame) -> DataFrame:
        """
        Remove words which are not meaningful as keywords / topics e.g. ministry positions that are held by a certain
        party, weekdays etc.
        :param dataframe: The dataframe to filter the nouns
        :return: The filtered data frame without blacklisted words
        """
        return dataframe["nouns"].apply(self._filter_words)

    def _filter_words(self, words: List[str]) -> List[str]:
        """
        Filter words (remove words contained in blacklist
        :param words: Words to filter
        :return: Filtered word list
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
