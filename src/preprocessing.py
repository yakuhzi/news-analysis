import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from pandas import DataFrame, Series
from pandas.core.common import SettingWithCopyWarning
from spacy.lang.de.stop_words import STOP_WORDS
from spacy_sentiws import spaCySentiWS
from tqdm import tqdm

from model.document_type import DocumentType
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

        self.negation_words = [
            "nicht",
            "nie",
            "kein",
            "weder",
            "nirgendwo",
            "ohne",
            "selten",
            "kaum",
            "trotz",
            "obwohl",
        ]
        self.negation_pattern = re.compile("|".join(self.negation_words))

    def get_articles(self, df_articles: DataFrame, overwrite: bool = False) -> DataFrame:
        return self._get_preprocessed_df("articles", df_articles, DocumentType.ARTICLE, overwrite)

    def get_paragraphs(self, df_articles: DataFrame, overwrite: bool = False) -> DataFrame:
        return self._get_preprocessed_df("paragraphs", df_articles, DocumentType.PARAGRAPH, overwrite)

    def get_titles(self, df_articles: DataFrame, overwrite: bool = False) -> DataFrame:
        return self._get_preprocessed_df("titles", df_articles, DocumentType.TITLE, overwrite)

    def _get_preprocessed_df(
        self, preprocessed_filename: str, articles: DataFrame, document_type: DocumentType, overwrite: bool
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
        json_path = "src/output/" + preprocessed_filename + ".json"

        if Path(json_path).exists() and not overwrite:
            return Reader.read_json_to_df_default(json_path)

        if document_type.value == DocumentType.ARTICLE.value:
            df_preprocessed = self._apply_preprocessing(articles)
        elif document_type.value == DocumentType.PARAGRAPH.value:
            df_preprocessed = self._preprocess_paragraphs(articles)
        else:
            articles = articles[["title", "media"]].rename(columns={"title": "text"})
            df_preprocessed = self._apply_preprocessing(articles, False)

        Writer.write_articles(df_preprocessed, preprocessed_filename)
        return df_preprocessed

    def _apply_preprocessing(self, dataframe: DataFrame, remove_rows_without_parties: bool = True) -> DataFrame:
        print("Start of preprocessing")
        start_time = time.time()

        # Copy original dataframe
        df_preprocessed = dataframe.copy()

        # Convert string date into datetime
        if "date" in df_preprocessed:
            df_preprocessed["date"].astype("datetime64[ns]")

        # Remove direct quotiations
        df_preprocessed["text"] = self._remove_direct_quotations(df_preprocessed["text"])

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

        # Get parties
        df_preprocessed["parties"] = self._get_parties(df_preprocessed["organizations"])

        # Remove rows with no parties
        if remove_rows_without_parties:
            df_preprocessed = self._remove_rows_without_parties(df_preprocessed)

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

        if remove_rows_without_parties:
            print("Number of documents after filtering: {}".format(len(df_preprocessed)))

        return df_preprocessed

    def _preprocess_paragraphs(self, df_articles: DataFrame) -> DataFrame:
        # Split articles into paragraphs by splitting at newlines
        paragraphs = list(map(lambda text: text.replace("\n+", "\n").split("\n"), df_articles["text"]))
        flat_list = [(index, item) for index, sublist in enumerate(paragraphs) for item in sublist]
        index, texts = zip(*flat_list)

        # Store paragraphs with the original article index and media
        dataframe = DataFrame(columns=["article_index", "text", "media", "date"])
        dataframe["article_index"] = index
        dataframe["text"] = texts
        dataframe["media"] = dataframe["article_index"].apply(lambda index: df_articles["media"][index])
        dataframe["date"] = dataframe["article_index"].apply(lambda index: df_articles["date"][index])
        dataframe["date"] = dataframe["date"].apply(lambda date: date.split("T")[0])
        dataframe["date"] = dataframe["date"].replace(r"^\s*$", np.nan, regex=True)

        return self._apply_preprocessing(dataframe)

    def _remove_direct_quotations(self, text_series: Series) -> Series:
        return text_series.str.replace(r'"(.*?)"', "", regex=True).str.strip()

    def _remove_special_characters(self, text_series: Series) -> Series:
        return (
            text_series.str.replace(r"[^A-Za-z0-9äöüÄÖÜß\-]", " ", regex=True)
            .str.replace(r" - ", " ", regex=False)
            .str.replace(r" +", " ", regex=True)
            .str.strip()
        )

    def _remove_stopwords(self, text_series: Series) -> Series:
        tqdm.pandas(desc="Remove stopwords")
        return text_series.progress_apply(
            lambda row: " ".join(word for word in row.split() if word not in self.stopwords)
        )

    def _tokenization(self, text_series: Series) -> Series:
        tqdm.pandas(desc="Tokenization")
        return text_series.progress_apply(lambda row: self.nlp(row))

    def _pos_tagging(self, token_series: Series) -> Series:
        tqdm.pandas(desc="POS Tagging")
        return token_series.progress_apply(lambda doc: [token.tag_ for token in doc])

    def _lemmatizing(self, token_series: Series) -> Series:
        tqdm.pandas(desc="Lemmatization")
        return token_series.progress_apply(lambda doc: [token.lemma_.lower() for token in doc])

    def _get_nouns(self, token_series: Series) -> Series:
        tqdm.pandas(desc="Get nouns")
        return token_series.progress_apply(lambda doc: [token.lemma_.lower() for token in doc if token.tag_ == "NN"])

    def _tag_persons(self, token_series: Series) -> Series:
        tqdm.pandas(desc="Tag persons")
        return token_series.progress_apply(
            lambda doc: list(set([entity.text for entity in doc.ents if entity.label_ == "PER"]))
        )

    def _tag_organizations(self, token_series: Series) -> Series:
        tqdm.pandas(desc="Tag organizations")
        return token_series.progress_apply(
            lambda doc: list(set([entity.text for entity in doc.ents if entity.label_ == "ORG"]))
        )

    def _remove_entities(self, token_series: Series) -> Series:
        tqdm.pandas(desc="Remove entities")
        return token_series.progress_apply(lambda doc: [token for token in doc if not token.ent_type_])

    def _get_parties(self, organization_series: Series) -> Series:
        tqdm.pandas(desc="Get parties")
        organization_series = organization_series.apply(lambda row: [x.lower() for x in row])
        return organization_series.progress_apply(
            lambda row: [party for party, synonyms in self.parties.items() if any(x in synonyms for x in row)]
        )

    def _remove_rows_without_parties(self, dataframe: DataFrame) -> DataFrame:
        tqdm.pandas(desc="Remove rows without parties")
        return dataframe.loc[np.array(list(map(len, dataframe.parties.values))) > 0]

    def determine_sentiment_polarity(self, token_series: Series) -> Series:
        tqdm.pandas(desc="Determine sentiment polarity")
        return token_series.progress_apply(lambda doc: [token._.sentiws for token in doc])

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
