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

    def set_active_media(self, media_list: list):
        self.df_paragraphs = self.df_paragraphs[self.df_paragraphs["media"].isin(media_list)]

    def get_term_weight_tuples(self, parties: list = None) -> DataFrame:
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

        terms = []

        for party in parties:
            terms += self._get_top_words(party, vectorizer, transformer)

        terms = list(set(terms))
        tuples = []

        for party in parties:
            for term in terms:
                term_count = self._get_term_count(self.df_paragraphs, party, term)
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

    def _get_top_words(self, party: str, vectorizer: CountVectorizer, transformer: TfidfTransformer) -> List[str]:
        paragraphs = self._get_party_paragraphs(self.df_paragraphs, party)
        paragraphs["nouns"] = self._remove_party_positions(paragraphs)

        nouns = paragraphs["nouns"].to_string()

        tf_idf_vector = transformer.transform(vectorizer.transform([nouns]))
        weights = np.asarray(tf_idf_vector.mean(axis=0)).ravel().tolist()

        df_weights = DataFrame({"term": vectorizer.get_feature_names(), "TF_IDF": weights})
        df_weights = df_weights.sort_values("TF_IDF", ascending=False).reset_index(drop=False)

        top_terms = df_weights["term"].tolist()[:3]
        print("Top 3 words of {}: {}".format(party, top_terms))
        return top_terms

    def _get_term_count(self, df, party: str, term: str) -> int:
        paragraphs = self._get_party_paragraphs(df, party)
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

    def _get_party_paragraphs(self, dataframe: DataFrame, party: str) -> DataFrame:
        return dataframe[dataframe.apply(lambda row: len(row["parties"]) == 1 and party in row["parties"], axis=1)]

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
