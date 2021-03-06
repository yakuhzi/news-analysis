import re
import time
import warnings
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import spacy
from pandas import DataFrame, Series
from pandas.core.common import SettingWithCopyWarning
from spacy.lang.de.stop_words import STOP_WORDS
from spacy_sentiws import spaCySentiWS
from textblob_de import TextBlobDE
from tqdm import tqdm

from model.document_type import DocumentType
from model.filter_type import FilterType
from utils.reader import Reader

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class Preprocessing:
    def __init__(self):
        self.stopwords = spacy.lang.de.stop_words.STOP_WORDS

        # de_core_news_lg had the best score for entity recognition and syntax accuracy in german according to spacy.
        # for more information, see https://spacy.io/models/de#de_core_news_lg
        self.nlp = None
        self.sentiws = None

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
        return self._get_preprocessed_df("paragraphs", df_articles, DocumentType.PARAGRAPH, overwrite)

    def get_titles(self, df_articles: DataFrame, overwrite: bool = False) -> DataFrame:
        """
        Helper function to get a preprocessed dataframe with the titles of the articles.
        :param df_articles: dataframe with the text to preprocess
        :param overwrite: determines if the previous data is allowed to be overwritten. Default is False.
        :return: preprocessed dataframe, which is result of method _get_preprocessed_df
        """
        return self._get_preprocessed_df("titles", df_articles, DocumentType.TITLE, overwrite)

    def _get_preprocessed_df(
        self, preprocessed_filename: str, df_articles: DataFrame, document_type: DocumentType, overwrite: bool
    ) -> DataFrame:
        """
        Helper function to get the preprocessed pandas dataframe. If the preprocessing already was done ones (JSON files
        exist) the tagging is not done again but the json files with the perprocessing are read into a pandas dataframe.
        If preprocessing is proceeded, the result will be stored in a json file. According to the document type, a
        different preprocessing is done.
        :param preprocessed_filename: Name of json file to store/ read the results of preprocessing.
        :param df_articles: Dataframe with the text to preprocess, if the data still needs to be preprocessed.
        :param document_type: Type of the document that is going to be preprocessed.
        :param overwrite: Determines if the previous data is allowed to be overwritten.
        :return: df_preprocessed: Pandas dataframe of the preprocessed input.
        """
        json_path = "src/output/" + preprocessed_filename + ".json"

        if Path(json_path).exists() and not overwrite:
            return Reader.read_json_to_df_default(json_path)

        if document_type.value == DocumentType.ARTICLE.value:
            df_preprocessed = self._apply_preprocessing(df_articles, document_type, FilterType.PARTIES)
        elif document_type.value == DocumentType.PARAGRAPH.value:
            df_preprocessed = self._preprocess_paragraphs(df_articles)
        else:
            df_articles = df_articles[["title", "media"]].rename(columns={"title": "text"})
            df_preprocessed = self._apply_preprocessing(df_articles, document_type, FilterType.NONE)

        return df_preprocessed

    def _apply_preprocessing(
        self, dataframe: DataFrame, document_type: DocumentType, filter_type: FilterType
    ) -> DataFrame:
        """
        Helper function responsible for applying preprocessing steps in correct order.
        :param dataframe: data that needs to be preprocessed.
        :param document_type: Type of the document that is going to be preprocessed.
        :param filter_type: Specifies if documents with no parties or multiple parties should be removed.
        :return: Preprocessed dataframe.
        """
        self.nlp = spacy.load("de_core_news_lg", disable=["parser"])
        self.sentiws = spaCySentiWS(sentiws_path="src/data/sentiws/")
        self.nlp.add_pipe(self.sentiws)

        nltk.download("punkt")

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

        # Remove rows with quotations if document is a paragraph
        if document_type.value == DocumentType.PARAGRAPH.value:
            df_preprocessed = self._remove_quotations_rows(df_preprocessed)

        # Remove special characters
        df_preprocessed["text"] = self._remove_special_characters(df_preprocessed["text"])

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

        # Sentiment polarity sentiws
        df_preprocessed["polarity"] = self._determine_polarity_sentiws(df_preprocessed["text"])

        # Sentiment polarity TextBlob
        df_preprocessed["polarity_textblob"] = self._determine_polarity_textblob(df_preprocessed["original_text"])

        # POS tagging
        df_preprocessed["pos_tags"] = self._pos_tagging(df_preprocessed["text"])

        # Get nouns
        df_preprocessed["nouns"] = self._get_nouns(df_preprocessed["text"])

        # Lemmatization
        df_preprocessed["text"] = self._lemmatizing(df_preprocessed["text"])

        # Negation handling
        df_preprocessed = self._negation_handling(df_preprocessed)

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

        return self._apply_preprocessing(dataframe, DocumentType.PARAGRAPH, FilterType.PARTIES)

    def _remove_direct_quotations(self, text_series: Series) -> Series:
        """
        removes direct quotations from the text in the series
        :param text_series: series that contains the text which needs to be preprocessed
        :return: series without direct quotations
        """
        return text_series.str.replace(r'"(.*?)"', "", regex=True).str.strip()

    def _remove_quotations_rows(self, dataframe: DataFrame) -> DataFrame:
        """
        removes dataframe entries that contain direct quotes
        :param dataframe: dataframe that contains the text which needs to be preprocessed
        :return: dataframe without entries containing direct quotation
        """
        return dataframe.loc[dataframe["text"].str.contains(r'["„“]') == 0]

    def _remove_special_characters(self, text_series: Series) -> Series:
        """
        removes special characters, e.g.: punctuation marks. Special characters which are essential
        for the german language are kept
        :param text_series: series, where removing special characters is performed on
        :return: series, without special characters
        """
        return (
            text_series.str.replace(r"[^A-Za-z0-9äöüÄÖÜß\-]", " ", regex=True)
            .str.replace(r" - ", " ", regex=False)
            .str.replace(r" +", " ", regex=True)
            .str.strip()
        )

    def _remove_stopwords(self, text_series: Series) -> Series:
        """
        removes stopwords
        :param text_series: series, where stopword removal is performed on
        :return: series, without stopwords
        """
        tqdm.pandas(desc="Remove stopwords")
        return text_series.progress_apply(
            lambda row: " ".join(word for word in row.split() if word not in self.stopwords)
        )

    def _tokenization(self, text_series: Series) -> Series:
        """
        tokenization is performed on the incoming series
        :param text_series:  series, containing the text where the tokenization is performed on
        :return: series, having the text transformed into tokens
        """
        tqdm.pandas(desc="Tokenization")
        return text_series.progress_apply(lambda row: self.nlp(row))

    def _pos_tagging(self, token_series: Series) -> Series:
        """
        POS-Tagging is performed on the incoming series
        :param token_series: series, containing text for pos-tagging
        :return: series, containing pos tags for each word in row
        """
        tqdm.pandas(desc="POS Tagging")
        return token_series.progress_apply(lambda doc: [token.tag_ for token in doc])

    def _lemmatizing(self, token_series: Series) -> Series:
        """
        Performs lemmatizing on incoming series
        :param token_series: series, containing the tokens where lemmatizing should be performed on
        :return: series, containing lammatized tags for each word in row
        """
        tqdm.pandas(desc="Lemmatization")
        return token_series.progress_apply(lambda doc: [token.lemma_.lower() for token in doc])

    def _get_nouns(self, token_series: Series) -> Series:
        """
        filters nouns based in pos_tagging
        :param token_series: series, containing pos tags
        :return: series, containing only nouns
        """
        tqdm.pandas(desc="Get nouns")
        return token_series.progress_apply(lambda doc: [token.lemma_.lower() for token in doc if token.tag_ == "NN"])

    def _tag_persons(self, token_series: Series) -> Series:
        """
        filters persons out of token series
        :param token_series: series, containing text where persons should be filtered
        :return: series, containing persons for each row
        """
        tqdm.pandas(desc="Tag persons")
        return token_series.progress_apply(
            lambda doc: list(set([entity.text for entity in doc.ents if entity.label_ == "PER"]))
        )

    def _tag_organizations(self, token_series: Series) -> Series:
        """
        filters organizations out of token series
        :param token_series: series, containing text where organizations should be filtered
        :return: series, containing organizations for each row
        """
        tqdm.pandas(desc="Tag organizations")
        return token_series.progress_apply(
            lambda doc: list(set([entity.text for entity in doc.ents if entity.label_ == "ORG"]))
        )

    def _get_parties(self, organization_series: Series) -> Series:
        """
        extract parties, defined in dictionary above, from tests
        :param organization_series: series, containing the previously found organizations
        :return: series, containing parties extracted from organizations
        """
        tqdm.pandas(desc="Get parties")
        organization_series = organization_series.apply(lambda row: [x.lower() for x in row])
        return organization_series.progress_apply(
            lambda row: [party for party, synonyms in self.parties.items() if any(x in synonyms for x in row)]
        )

    def _remove_rows_without_parties(self, dataframe: DataFrame) -> DataFrame:
        """
        all rows, that do not contain parties are removed
        :param dataframe: dataframe, containing previously preprocessed data
        :return: dataframe, containing previously preprocessed data without data not containing parties
        """
        return dataframe.loc[np.array(list(map(len, dataframe.parties.values))) > 0]

    def _keep_rows_with_one_party(self, dataframe: DataFrame) -> DataFrame:
        """
        removes data that contain more or less than one party.
        :param dataframe: dataframe, containing previously preprocessed data
        :return: dataframe, containing previously preprocessed data without data not containing parties
        """
        return dataframe.loc[np.array(list(map(len, dataframe.parties.values))) == 1]

    def _determine_polarity_sentiws(self, token_series: Series) -> Series:
        """
        for each word in a row of the series, the polarity is calculated with sentiws.
        :param token_series: series, containing the tokens which need to be calculated
        :return: series, containing rows with the polarities for each token
        """
        tqdm.pandas(desc="Determine sentiment polarity with SentiWS")
        return token_series.progress_apply(lambda doc: [token._.sentiws for token in doc])

    def _determine_polarity_textblob(self, text_series: Series) -> Series:
        """
        for each paragraph (row in a series) the polarity is calculated with textblob
        :param text_series: series, containing the text where the polarity needs to be determined
        :return: series, containing rows with the polarity for the corresponding text
        """
        tqdm.pandas(desc="Determine sentiment polarity with TextBlob")
        return text_series.progress_apply(lambda doc: TextBlobDE(doc).sentiment[0])

    def _negation_handling(self, df_preprocessed: DataFrame) -> DataFrame:
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
