import re
from typing import List

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class KeywordExtraction:
    def __init__(self, df_paragraphs):
        self.df_paragraphs = df_paragraphs

    def set_data(self, df_paragraphs):
        self.df_paragraphs = df_paragraphs

    def set_active_media(self, media_list: List[str]):
        self.df_paragraphs = self.df_paragraphs[self.df_paragraphs["media"].isin(media_list)]

    def get_term_weight_tuples(
        self, by_party: bool, all_terms: bool = False, parties: List[str] = None, media: List[str] = None
    ) -> DataFrame:
        # Get nouns from dataframe
        nouns = self.df_paragraphs["nouns"].apply(lambda row: " ".join(row))

        # Vectorize character_words
        vectorizer = CountVectorizer(min_df=5, token_pattern=r"(?u)\b[a-zA-Z0-9_\-][a-zA-Z0-9_\-]+\b")
        count_vectorized = vectorizer.fit_transform(nouns)

        # Apply tf-idf to count_vectorized
        transformer = TfidfTransformer(smooth_idf=True, use_idf=True)

        # Generate tf-idf for the given document
        transformer.fit(count_vectorized)

        if by_party and parties is None or all_terms:
            parties = ["CDU", "CSU", "SPD", "FDP", "AfD", "Grüne", "Linke"]

        if not by_party and media is None or all_terms:
            media = ["Bild", "Tagesschau", "TAZ"]

        terms = []

        for party in parties:
            terms += self._get_top_words(True, party, vectorizer, transformer)

        for m in media:
            terms += self._get_top_words(False, m, vectorizer, transformer)

        terms = list(set(terms))
        tuples = []

        for party_or_media in parties if by_party else media:
            for term in terms:
                term_count = self._get_term_count(by_party, party_or_media, term)
                color = "r" if party_or_media == "Bild" else "g"
                tuples.append((party_or_media, term, term_count, color))

        return DataFrame(tuples, columns=["party" if by_party else "media", "term", "weight", "color"])

    def _get_top_words(
        self, by_party: bool, party_or_media: str, vectorizer: CountVectorizer, transformer: TfidfTransformer
    ) -> List[str]:
        if by_party:
            paragraphs = self._get_party_paragraphs(self.df_paragraphs, party_or_media)
        else:
            paragraphs = self._get_media_paragraphs(self.df_paragraphs, party_or_media)

        paragraphs["nouns"] = self._remove_party_positions(paragraphs)
        nouns = paragraphs["nouns"].to_string()

        tf_idf_vector = transformer.transform(vectorizer.transform([nouns]))
        weights = np.asarray(tf_idf_vector.mean(axis=0)).ravel().tolist()

        df_weights = DataFrame({"term": vectorizer.get_feature_names(), "TF_IDF": weights})
        df_weights = df_weights.sort_values("TF_IDF", ascending=False).reset_index(drop=False)

        top_terms = df_weights["term"].tolist()[:3]
        print("Top 3 words of {}: {}".format(party_or_media, top_terms))
        return top_terms

    def _get_term_count(self, by_party: bool, party_or_media: str, term: str) -> int:
        if by_party:
            paragraphs = self._get_party_paragraphs(self.df_paragraphs, party_or_media)
        else:
            paragraphs = self._get_media_paragraphs(self.df_paragraphs, party_or_media)

        return paragraphs["nouns"].apply(lambda row: row.count(term)).sum()

    def get_graph(self, df_term_weights: DataFrame):
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
            color="r",
        )

        graph.add_weighted_edges_from(
            [(row["media"], row["term"], row["weight"]) for idx, row in df_media_weights.iterrows()],
            weight="weight",
            color="g",
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
        return dataframe[dataframe.apply(lambda row: len(row["parties"]) == 1 and party in row["parties"], axis=1)]

    def _get_media_paragraphs(self, dataframe: DataFrame, media: str) -> DataFrame:
        return dataframe[dataframe.apply(lambda row: media == row["media"], axis=1)]

    def _remove_party_positions(self, dataframe: DataFrame) -> DataFrame:
        return dataframe["nouns"].apply(self._filter_words)

    def _filter_words(self, words: List[str]) -> List[str]:
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
