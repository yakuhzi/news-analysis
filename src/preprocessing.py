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
from model.filter_type import FilterType
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
        """
        Helper function to get a dataframe containing the preprocessed articles.
        :param df_articles: dataframe with the text to preprocess
        :param overwrite: determines if the previous data is allowed to be overwritten. Default is False.
        :return: preprocessed dataframe with preprocessed articles, which is result of method _get_preprocessed_df
        """
        return self._get_preprocessed_df("articles", df_articles, DocumentType.ARTICLE, overwrite)

    def get_paragraphs(self, df_articles: DataFrame, overwrite: bool = False) -> DataFrame:
        """
        Helper function to get a preprocessed dataframe with the paragraphs of the articles.
        :param df_articles: dataframe with the text to preprocess
        :param overwrite: determines if the previous data is allowed to be overwritten. Default is False.
        :return: preprocessed dataframe with preprocessed paragraphs, which is result of method _get_preprocessed_df
        """
        return self._get_preprocessed_df("labeled_paragraphs", df_articles, DocumentType.PARAGRAPH, overwrite)

    def get_titles(self, df_articles: DataFrame, overwrite: bool = False) -> DataFrame:
        """
        Helper function to get a preprocessed dataframe with the titles of the articles.
        :param df_articles: dataframe with the text to preprocess
        :param overwrite: determines if the previous data is allowed to be overwritten. Default is False.
        :return: preprocessed dataframe, which is result of method _get_preprocessed_df
        """
        return self._get_preprocessed_df("titles", df_articles, DocumentType.TITLE, overwrite)

    def _get_preprocessed_df(
        self, preprocessed_filename: str, articles: DataFrame, document_type: DocumentType, overwrite: bool
    ) -> DataFrame:
        """
        Helper function to get the preprocessed pandas dataframe. If the preprocessing already was done ones (JSON files
        exist) the tagging is not done again but the json files with the perprocessing are read into a pandas dataframe.
        If preprocessing is proceeded, the result will be stored in a json file. According to the document type, a
        different preprocessing is done.
        :param preprocessed_filename: Name of json file to store/ read the results of preprocessing.
        :param articles: dataframe with the text to preprocess, if the data still needs to be preprocessed.
        :param document_type: type of the document that is going to be preprocessed.
        :param overwrite: determines if the previous data is allowed to be overwritten.
        :return: df_preprocessed: Pandas data frame of the preprocessed input.
        """
        json_path = "src/output/" + preprocessed_filename + ".json"

        if Path(json_path).exists() and not overwrite:
            return Reader.read_json_to_df_default(json_path)

        if document_type.value == DocumentType.ARTICLE.value:
            df_preprocessed = self._apply_preprocessing(articles, FilterType.PARTIES)
        elif document_type.value == DocumentType.PARAGRAPH.value:
            df_preprocessed = self._preprocess_paragraphs(articles)
        else:
            articles = articles[["title", "media"]].rename(columns={"title": "text"})
            df_preprocessed = self._apply_preprocessing(articles, FilterType.NONE)

        Writer.write_dataframe(df_preprocessed, preprocessed_filename)
        return df_preprocessed

    def _apply_preprocessing(self, dataframe: DataFrame, filter_type: FilterType) -> DataFrame:
        """
        Helper function responsible for applying preprocessing steps in correct order.
        :param dataframe: data that needs to be preprocessed.
        :param remove_rows_without_parties: determines if rows, that do not contain information about parties are deleted.
        :return: preprcessed dataframe.
        """
        print("Start of preprocessing")
        start_time = time.time()

        # Copy original dataframe
        df_preprocessed = dataframe.copy()

        # Convert string date into datetime
        if "date" in df_preprocessed:
            df_preprocessed["date"] = df_preprocessed["date"].apply(lambda date: date.split("T")[0])
            df_preprocessed["date"] = df_preprocessed["date"].replace(r"^\s*$", np.nan, regex=True)
            df_preprocessed["date"].astype("datetime64[ns]")

        df_preprocessed["original_text"] = df_preprocessed["text"]

        # Remove rows with quotations
        df_preprocessed = self._remove_quotations_rows(df_preprocessed)

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
        if filter_type.value == FilterType.PARTIES.value:
            df_preprocessed = self._remove_rows_without_parties(df_preprocessed)

        # Remove rows with no parties or more than one party
        if filter_type.value == FilterType.SINGLE_PARTY.value:
            df_preprocessed = self._keep_rows_with_one_party(df_preprocessed)

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

        if filter_type.value != FilterType.NONE.value:
            print("Number of documents after filtering: {}".format(len(df_preprocessed)))

        return df_preprocessed

    def _preprocess_paragraphs(self, df_articles: DataFrame) -> DataFrame:
        """
        Helper function that splits up paragraphs and stores them with the original article in a dataframe. After that
        the original preprocessing is done.
        :param df_articles: dataframe with articles that need to be split up in paragraphs.
        :return: dataframe with preprocessed articles, split up by paragraph.
        """
        # Split articles into paragraphs by splitting at newlines
        paragraphs = list(map(lambda text: text.replace("\n+", "\n").split("\n"), df_articles["text"]))
        flat_list = [(index, item) for index, sublist in enumerate(paragraphs) for item in sublist]
        index, texts = zip(*flat_list)

        # Store paragraphs with the original article index and media
        dataframe = DataFrame(columns=["article_index", "title", "text", "media", "date"])
        dataframe["article_index"] = index
        dataframe["title"] = dataframe["article_index"].apply(lambda index: df_articles["title"][index])
        dataframe["text"] = texts
        dataframe["media"] = dataframe["article_index"].apply(lambda index: df_articles["media"][index])
        dataframe["date"] = dataframe["article_index"].apply(lambda index: df_articles["date"][index])

        return self._apply_preprocessing(dataframe, FilterType.PARTIES)

    def _remove_direct_quotations(self, text_series: Series) -> Series:
        return text_series.str.replace(r'"(.*?)"', "", regex=True).str.strip()

    def _remove_quotations_rows(self, dataframe: DataFrame) -> DataFrame:
        return dataframe.loc[dataframe["text"].str.contains(r'["„“]') is False]

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
        return dataframe.loc[np.array(list(map(len, dataframe.parties.values))) > 0]

    def _keep_rows_with_one_party(self, dataframe: DataFrame) -> DataFrame:
        return dataframe.loc[np.array(list(map(len, dataframe.parties.values))) == 1]

    def determine_sentiment_polarity(self, token_series: Series) -> Series:
        tqdm.pandas(desc="Determine sentiment polarity")
        return token_series.progress_apply(lambda doc: [token._.sentiws for token in doc])

    def negation_handling(self, df_preprocessed: DataFrame) -> DataFrame:
        """
        Checks if 4 tokens before or after sentiws assigned a polarity score a negation word can be found. If this is the
        case, the polarity is inverted.
        :param df_preprocessed: Dataframe containing scores from sentiws for each word.
        :return: Dataframe with inverted scores if a negation word could be found.
        """
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
