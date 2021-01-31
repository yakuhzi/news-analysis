import re
import time
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import spacy
from pandas import DataFrame, Series
from pandas.core.common import SettingWithCopyWarning
from spacy.lang.de.stop_words import STOP_WORDS
from spacy_sentiws import spaCySentiWS
from utils.document_type import DocumentType
from utils.reader import Reader
from utils.writer import Writer

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class Preprocessing:
    def __init__(self):
        self.stopwords = spacy.lang.de.stop_words.STOP_WORDS

        # de_core_news_lg had the best score for entity recognition and syntax accuracy in german according to spacy.
        # for more information, see https://spacy.io/models/de#de_core_news_lg
        self.nlp = spacy.load("de_core_news_lg", disable=["parser"])
        self.sentiws = spaCySentiWS(sentiws_path="src/data/sentiws/")
        self.nlp.add_pipe(self.sentiws)

        self.parties = {
            "CDU": ["cdu", "union"],
            "CSU": ["csu"],
            "SPD": ["spd", "sozialdemokraten"],
            "Grüne": [
                "grüne",
                "grünen",
                "die grüne",
                "die grünen",
                "den grünen",
                "bündnis90 die grünen",
            ],
            "FDP": [
                "fdp",
                "liberalen",
                "freien demokrate",
                "freie demokratische partei",
            ],
            "AfD": ["afd", "alternative für deutschland"],
            "Linke": ["linke", "die linke", "den linken"],
        }

        self.negation_words = ["nicht", "kein", "nein"]
        self.negation_pattern = re.compile("nicht|kein|nein")

    def get_articles(self, reader: Reader) -> Tuple[DataFrame, DataFrame, DataFrame]:
        df_bild_preprocessed = self._get_preprocessed_df(
            "bild_preprocessed", reader.df_bild_articles, DocumentType.ARTICLE
        )

        df_tagesschau_preprocessed = self._get_preprocessed_df(
            "tagesschau_preprocessed", reader.df_tagesschau_articles, DocumentType.ARTICLE
        )

        df_taz_preprocessed = self._get_preprocessed_df(
            "taz_preprocessed", reader.df_taz_articles, DocumentType.ARTICLE
        )

        return df_bild_preprocessed, df_tagesschau_preprocessed, df_taz_preprocessed

    def get_titles(self, reader: Reader) -> Tuple[DataFrame, DataFrame, DataFrame]:
        df_bild_preprocessed_titles = self._get_preprocessed_df(
            "bild_titles", reader.df_bild_articles, DocumentType.TITLE
        )

        df_tagesschau_preprocessed_titles = self._get_preprocessed_df(
            "tagesschau_titles", reader.df_tagesschau_articles, DocumentType.TITLE
        )

        df_taz_preprocessed_titles = self._get_preprocessed_df("taz_titles", reader.df_taz_articles, DocumentType.TITLE)

        return df_bild_preprocessed_titles, df_tagesschau_preprocessed_titles, df_taz_preprocessed_titles

    def get_paragraphs(self, reader: Reader) -> Tuple[DataFrame, DataFrame, DataFrame]:
        df_bild_preprocessed_paragraphs = self._get_preprocessed_df(
            "bild_paragraphs",
            reader.df_bild_articles,
            DocumentType.PARAGRAPH,
        )

        df_tagesschau_preprocessed_paragraphs = self._get_preprocessed_df(
            "tagesschau_paragraphs", reader.df_tagesschau_articles, DocumentType.PARAGRAPH
        )

        df_taz_preprocessed_paragraphs = self._get_preprocessed_df(
            "taz_paragraphs", reader.df_taz_articles, DocumentType.PARAGRAPH
        )

        return df_bild_preprocessed_paragraphs, df_tagesschau_preprocessed_paragraphs, df_taz_preprocessed_paragraphs

    def _get_preprocessed_df(
        self,
        preprocessed_file: str,
        articles: DataFrame,
        document_type: DocumentType,
    ) -> DataFrame:
        """
        Helper function to get the preprocessed pandas dataframe. If the preprocessing already was done ones (JSON files
        exist) the tagging is not done again but the json files with the perprocessing are read into a pandas data frame.
        If preprocessing is proceeded, the result will be stored in a json file.

        Arguments:
        - preprocessed_json_file: Name of json file to store/ read the results of preprocessing.
        - df_to_preprocess: data frame with the text to preprocess, if the data still needs to be preprocessed

        Return:
        - df_preprocessed: Pandas data frame of the preprocessed input
        """
        json_path = "src/output/" + preprocessed_file + ".json"

        if not Path(json_path).exists():
            df_preprocessed: DataFrame

            if document_type.value == DocumentType.TITLE.value:
                articles = articles[["title"]].rename(columns={"title": "text"})
                df_preprocessed = self._apply_preprocessing(articles, False)
            elif document_type.value == DocumentType.PARAGRAPH.value:
                df_preprocessed = self._preprocess_paragraphs(articles)
            else:
                df_preprocessed = self._apply_preprocessing(articles)

            Writer.write_articles(df_preprocessed, preprocessed_file)
        else:
            df_preprocessed = Reader.read_json_to_df_default(json_path)

        return df_preprocessed

    def _apply_preprocessing(self, dataframe: DataFrame, remove_rows_without_parties: bool = True) -> DataFrame:
        print("Start of preprocessing")
        start_time = time.time()

        # Copy original dataframe
        df_preprocessed = dataframe.copy()

        # Remove special characters
        df_preprocessed["text"] = self._remove_special_characters(df_preprocessed["text"])

        # Stop word removal
        # df_preprocessed["text"] = self._remove_stopwords(df_preprocessed["text"])

        # Tokenization
        df_preprocessed["text"] = self._tokenization(df_preprocessed["text"])

        # Get persons
        df_preprocessed["persons"] = self._tag_persons(df_preprocessed["text"])

        # Get organizations
        df_preprocessed["organizations"] = self._tag_organizations(df_preprocessed["text"])

        # Remove rows with no parties
        if remove_rows_without_parties:
            df_preprocessed = self._remove_rows_without_parties(df_preprocessed)
            print("Number of documents after filtering: {}".format(len(df_preprocessed)))

        # Sentiment polarity
        df_preprocessed["polarity"] = self.determine_sentiment_polarity(df_preprocessed["text"])

        # POS tagging
        df_preprocessed["pos_tags"] = self._pos_tagging(df_preprocessed["text"])

        # Get nouns
        df_preprocessed["nouns"] = self._get_nouns(df_preprocessed["text"])

        # Lemmatization
        df_preprocessed["text"] = self._lemmatizing(df_preprocessed["text"])

        # Negation handling
        df_preprocessed = self.negation_handling(df_preprocessed)

        end_time = time.time()
        print("End of preprocessing after {} seconds".format(end_time - start_time))
        return df_preprocessed

    def _preprocess_paragraphs(self, articles) -> DataFrame:
        # Split articles into paragraphs by splitting at newlines
        paragraphs = list(map(lambda text: text.replace("\n*", "\n").split("\n"), articles["text"]))
        flat_list = [(index, item) for index, sublist in enumerate(paragraphs) for item in sublist]
        index, texts = zip(*flat_list)

        # Store paragraphs with the original article index
        dataframe = DataFrame(columns=["article_index", "text"])
        dataframe["article_index"] = index
        dataframe["text"] = texts

        return self._apply_preprocessing(dataframe)

    def _remove_special_characters(self, text_series: Series) -> Series:
        return (
            text_series.str.replace(r"[^A-Za-z0-9äöüÄÖÜß\-]", " ", regex=True)
            .str.replace(r" - ", "", regex=False)
            .str.replace(r" +", " ", regex=True)
        )

    def _remove_stopwords(self, text_series: Series) -> Series:
        return text_series.apply(lambda row: " ".join(word for word in row.split() if word not in self.stopwords))

    def _tokenization(self, text_series: Series) -> Series:
        return text_series.apply(lambda row: self.nlp(row))

    def _pos_tagging(self, token_series: Series) -> Series:
        return token_series.apply(lambda doc: [token.tag_ for token in doc])

    def _lemmatizing(self, token_series: Series) -> Series:
        return token_series.apply(lambda doc: [token.lemma_.lower() for token in doc])

    def _get_nouns(self, token_series: Series) -> Series:
        return token_series.apply(lambda doc: [token.lemma_.lower() for token in doc if token.tag_ == "NN"])

    def _tag_persons(self, token_series: Series) -> Series:
        return token_series.apply(lambda doc: list(set([entity.text for entity in doc.ents if entity.label_ == "PER"])))

    def _tag_organizations(self, token_series: Series) -> Series:
        return token_series.apply(lambda doc: list(set([entity.text for entity in doc.ents if entity.label_ == "ORG"])))

    def _remove_entities(self, token_series: Series) -> Series:
        return token_series.apply(lambda doc: [token for token in doc if not token.ent_type_])

    def _remove_rows_without_parties(self, dataframe: DataFrame) -> DataFrame:
        df_preprocessed = dataframe.apply(self._find_parties, axis=1)
        return df_preprocessed.loc[np.array(list(map(len, df_preprocessed.parties.values))) > 0]

    def _find_parties(self, row):
        organizations = [x.lower() for x in row["organizations"]]
        parties = []

        for organization in organizations:
            for key, value in self.parties.items():
                if organization in value and key not in parties:
                    parties.append(key)

        row["parties"] = parties
        return row

    def determine_sentiment_polarity(self, token_series: Series) -> Series:
        return token_series.apply(lambda doc: [token._.sentiws for token in doc])

    def negation_handling(self, df_preprocessed: DataFrame) -> DataFrame:
        polarity_array = df_preprocessed["polarity"].to_numpy()
        word_array = df_preprocessed["text"].to_numpy()

        for entry in range(len(polarity_array)):
            for i in range(len(polarity_array[entry])):
                if polarity_array[entry][i] is not None:
                    backward_window = i - 4
                    forward_window = i + 4

                    if backward_window < 0:
                        backward_window = 0
                    if forward_window >= len(polarity_array[entry]):
                        forward_window = len(polarity_array[entry])

                    words_in_window = word_array[entry][backward_window:forward_window]
                    words_in_window = " ".join(map(str, words_in_window))

                    if self.negation_pattern.search(words_in_window):
                        if polarity_array[entry][i] < 0:
                            polarity_array[entry][i] = abs(polarity_array[entry][i])
                        elif polarity_array[entry][i] > 0:
                            polarity_array[entry][i] = -polarity_array[entry][i]

        df_preprocessed.drop("polarity", inplace=True, axis=1)
        polarity_data = DataFrame(data=polarity_array, columns=["polarity"])
        df_preprocessed.reset_index(drop=True, inplace=True)
        polarity_data.reset_index(drop=True, inplace=True)
        result = pd.concat([df_preprocessed, polarity_data], axis=1)
        return result
