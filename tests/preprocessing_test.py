import os
import sys

testdir = os.path.dirname(__file__)
srcdir = "../src"
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

import unittest

import nltk
import spacy
from pandas import DataFrame, Series
from spacy_sentiws import spaCySentiWS

from src.preprocessing import Preprocessing


class NERTaggerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = spacy.load("de_core_news_lg", disable=["parser"])
        cls.preprocessing = Preprocessing()
        cls.preprocessing.nlp = spacy.load("de_core_news_lg", disable=["parser"])
        cls.preprocessing.sentiws = spaCySentiWS(sentiws_path="src/data/sentiws/")
        cls.preprocessing.nlp.add_pipe(cls.preprocessing.sentiws)

        nltk.download("punkt")

    def test_remove_quotations(self):
        text = 'Das ist ein Text mit direkter "Rede"'
        series = Series([text])
        text_without_quotation = self.preprocessing._remove_direct_quotations(series)[0]
        self.assertEqual(text_without_quotation, "Das ist ein Text mit direkter")

    def test_remove_special_characters(self):
        text = "Dies ist ein Text, der einige + Sonderzeichen enthält - diese sollten entfernt werden?!:)€"
        series = Series([text])
        text_without_special_characters = self.preprocessing._remove_special_characters(series)[0]
        self.assertEqual(
            text_without_special_characters,
            "Dies ist ein Text der einige Sonderzeichen enthält " "diese sollten entfernt werden",
        )

    def test_remove_stopwords(self):
        text = "Dies ist ein Text, der Stopwords enthält und Stopwords sind nicht wichtig und sollen entfernt werden."
        series = Series([text])
        text_without_stopwords = self.preprocessing._remove_stopwords(series)[0]
        self.assertEqual(text_without_stopwords, "Dies Text, Stopwords enthält " "Stopwords wichtig entfernt werden.")

    def test_tokenization(self):
        text = "Dieser Text soll in der Tokenization in Tokens unterteilt werden"
        series = Series([text])
        tokens = self.preprocessing._tokenization(series)[0]
        self.assertEqual(tokens[0].text, "Dieser")

    def test_pos_tagging(self):
        text = "In diesem Test soll das POS Tagging getestet werden."
        series = Series([text])
        tokens = self.preprocessing._tokenization(series)
        pos_tagged = self.preprocessing._pos_tagging(tokens)[0]
        self.assertEqual(pos_tagged, ["APPR", "PDAT", "NN", "VMFIN", "ART", "NN", "NE", "VVPP", "VAINF", "$."])

    def test_get_nouns(self):
        text = (
            "In diesem Text sollen die Substantive gefunden werden. Solche Wörter können für "
            "Themen in Artikeln interessant sein."
        )
        series = Series([text])
        tokens = self.preprocessing._tokenization(series)
        nouns = self.preprocessing._get_nouns(tokens)[0]
        self.assertEqual(nouns, ["text", "substantiv", "wort", "thema", "artikel"])

    def test_lemmatize(self):
        text = (
            "Von diesem Text sollen die Lemmata übrig bleiben. Das heißt, "
            "dass von Wörtern die Grundformen gefunden werden sollen."
        )
        series = Series([text])
        tokens = self.preprocessing._tokenization(series)
        lemmas = self.preprocessing._lemmatizing(tokens)[0]
        self.assertEqual(
            lemmas,
            [
                "von",
                "dies",
                "text",
                "sollen",
                "der",
                "lemmata",
                "übrig",
                "bleiben",
                ".",
                "der",
                "heißen",
                ",",
                "dass",
                "von",
                "wort",
                "der",
                "grundform",
                "finden",
                "werden",
                "sollen",
                ".",
            ],
        )  #

    def test_ner_tagging(self):
        text_to_tag = "Angela Merkel ist die deutsche Bundeskanzlerin und sie ist Mitglied der CDU."
        series = Series([self.nlp(text_to_tag)])

        persons = self.preprocessing._tag_persons(series).tolist()[0]
        organizations = self.preprocessing._tag_organizations(series).tolist()[0]

        self.assertEqual(persons, ["Angela Merkel"], "The person list should contain Angela Merkel")
        self.assertEqual(organizations, ["CDU"], "The organization list should contain CDU")

    def test_filter_ner_duplicates(self):
        text_to_tag = (
            "Die FDP ist eine Partei. Der Vorsitzende der FDP ist Christian Lindner. Die SPD ist eine andere Partei!"
        )
        series = Series([self.nlp(text_to_tag)])

        persons = self.preprocessing._tag_persons(series).tolist()[0]
        organizations = sorted(self.preprocessing._tag_organizations(series).tolist()[0])

        self.assertEqual(persons, ["Christian Lindner"], "The person list should contain Christian Lindner")
        self.assertEqual(organizations, ["FDP", "SPD"], "FDP should be filtered out of the organization list")

    def test_get_parties(self):
        text = (
            "Es gibt viele Parteien wie die CDU, die Grünen, die Liberalen, die Sozialdemokraten," " die AfD und andere"
        )
        series = Series([text])
        tokens = self.preprocessing._tokenization(series)
        organizations = self.preprocessing._tag_organizations(tokens)
        parties = self.preprocessing._get_parties(organizations)[0]
        self.assertEqual(parties, ["CDU", "SPD", "Grüne", "FDP", "AfD"])

    def test_remove_rows_without_parties(self):
        with_party = "In dieser Reihe kommt die CDU vor, sie sollte nicht gefiltert werden"
        without_party = "In dieser Reihe kommt keine Partei vor, sie sollte gefiltert werden"
        d = {"text": [with_party, without_party]}
        dataframe = DataFrame(data=d)
        dataframe["text"] = self.preprocessing._tokenization(dataframe["text"])
        dataframe["organizations"] = self.preprocessing._tag_organizations(dataframe["text"])
        dataframe["parties"] = self.preprocessing._get_parties(dataframe["organizations"])
        dataframe = self.preprocessing._remove_rows_without_parties(dataframe)
        self.assertEqual(dataframe.shape[0], 1)
        party_row_string = " ".join([token.text for token in dataframe["text"][0]])
        self.assertEqual(party_row_string, "In dieser Reihe kommt die CDU vor , sie sollte nicht gefiltert werden")

    def test_determine_polarity_sentiws(self):
        series = Series(["Heute ist ein schöner Tag", "Heute ist ein schlechter Tag"])
        tokens = self.preprocessing._tokenization(series)
        polarity = self.preprocessing._determine_polarity_sentiws(tokens)

        self.assertGreater(polarity[0][3], 0)
        self.assertLess(polarity[1][3], 0)
        self.assertIsNone(polarity[0][0])

    def test_determine_polarity_textblob(self):
        series = Series(["Heute ist ein schöner Tag", "Heute ist ein schlechter Tag"])
        polarity = self.preprocessing._determine_polarity_textblob(series)

        self.assertGreater(polarity[0], 0)
        self.assertLess(polarity[1], 0)

    def test_negation_handling(self):
        dataframe = DataFrame(data={"text": ["Heute ist kein schöner Tag"]})
        dataframe["text"] = self.preprocessing._tokenization(dataframe["text"])
        dataframe["polarity"] = self.preprocessing._determine_polarity_sentiws(dataframe["text"])
        dataframe = self.preprocessing._negation_handling(dataframe)
        polarity = dataframe["polarity"][0]
        self.assertLess(polarity[3], 0)
        self.assertIsNone(polarity[0])


if __name__ == "__main__":
    unittest.main()
