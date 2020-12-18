import itertools
from collections import defaultdict
from typing import Dict, List, Tuple

import nltk
import pandas as pd
import spacy
from spacy.lang.de.stop_words import STOP_WORDS


class PreprocessArticles:
    def __init__(self):
        self.nlp = None

    def lowercase_article(self, articles):
        articles["text"] = articles["text"].str.lower()

    def replace_new_line(self, df_preprocessed_articles):
        df_preprocessed_articles["text"] = df_preprocessed_articles["text"].str.replace(
            "\\n", " "
        )

    def remove_special_characters(self, df_preprocessed_articles):
        df_preprocessed_articles["text"] = df_preprocessed_articles["text"].str.replace(
            r"[^A-Za-z0-9äöüÄÖÜß\- ]", " "
        )

    def remove_stopwords(self, df_preprocessed_articles):
        stopwords = spacy.lang.de.stop_words.STOP_WORDS
        df_preprocessed_articles["text"] = df_preprocessed_articles["text"].apply(
            lambda words: " ".join(
                word for word in words.split() if word not in stopwords
            )
        )

    def tokenization(self, df_preprocessed_articles):
        df_preprocessed_articles["text"] = df_preprocessed_articles["text"].apply(
            lambda x: self.nlp(x)
        )

    def pos_tagging(self, df_preprocessed_articles):
        df_preprocessed_articles["pos_tags"] = df_preprocessed_articles["text"].apply(
            lambda row: [(word, word.tag_) for word in row]
        )

    def lemmatizing(self, df_preprocessed_articles):
        df_preprocessed_articles["lemma"] = df_preprocessed_articles["text"].apply(
            lambda row: [word.lemma_ for word in row]
        )

    def tag_dataframe(self, row: pd.Series) -> pd.Series:
        """
        Function to apply on Pandas data frame that it is tagged

        Arguments:
        - row: the current row of the data frame to be tagged

        Return:
        - row: Pandas series with the tagged text in colums 'persons' and 'rows'
        """
        persons, organizations = self.tag(row.text)
        # distance for persons can be bigger than parties, TODO: check if 3 is a good choice
        row["persons_ner"] = PreprocessArticles.filter_out_synonyms(persons, 3)
        # distance for parties can only be one because of acronyms (FDP, SPD,...)
        # TODO: what todo with CDU/CSU? treat as one party or two different
        row["organizations_ner"] = PreprocessArticles.filter_out_synonyms(
            organizations, 1
        )
        return row

    def tag(self, content: str) -> Tuple[List[str], List[str]]:
        """
        Searches for Names and Organizations in texts in order to identify relevant articles with political parties

        Arguments:
        - content: The text to search for the Named Entities

        Return:
        - person_list: List of recognized persons in the text.
        - organization_list: List of organizations in the text.
        """
        if self.nlp is None:
            self.nlp = spacy.load("de_core_news_lg", disable=["parser"])
        doc = self.nlp(content)
        #  search for persons and apply filter that only persons remain in list
        filtered_persons = filter(lambda entity: entity.label_ == "PER", doc.ents)
        person_list = list(map(lambda entity: entity.text, filtered_persons))
        #  search for organizations and apply filter that only persons remain in list
        filtered_organizations = filter(lambda entity: entity.label_ == "ORG", doc.ents)
        organization_list = list(
            map(lambda entity: entity.text, filtered_organizations)
        )
        return person_list, organization_list

    @staticmethod
    def calculate_word_distance(list_to_filter: List[str]) -> Dict[str, List]:
        """
        Calculates the distance between two words with the Levenshtein distance

        Arguments:
        - list_to_filter: the list with the words to calculate the distance

        Return:
        - similar_dict: dictionary with information about the distance
                structure key: word in list_to_filter
                          value: list [distance_to_other_word, other_word]
        """
        similar_dict = defaultdict(list)
        # go over all possible combinations between two words
        for a, b in itertools.combinations(list_to_filter, 2):
            dist = nltk.edit_distance(a, b)
            similar_dict[a].append((dist, b))
        return similar_dict

    @staticmethod
    def build_list_without_synonyms(
        list_to_filter: List[str],
        similar_dict: Dict[str, List],
        biggest_allowed_distance: int,
    ) -> List[str]:
        """
        Filters out the words of the list that have a smaller distance than biggest_allowed_distance

        Arguments:
        - list_to_filter: the list where duplicates should be removed
        - similar_dict: dictionary with information about distances between words in list
        - biggest_allowed_distance: maximum distance between two words

        Return:
        - new_ner_list: list without synonyms
        """
        # empty list where to store all words that are no synonyms
        new_ner_list = []
        # list of lists with all synonyms to a word in the dict (key)
        overall_similar_list = []
        for key, value in similar_dict.items():
            similar_list = []
            # indicate if already a synonym for a key has been found
            found = False
            for val in value:
                if int(val[0]) <= biggest_allowed_distance:
                    # if no synonym has been found yet, also the key itself needs to be put in similarity list
                    # otherwise the key is already in similarity list and only the word that is similar to the key
                    # needs to be stored
                    if not found:
                        similar_list.append(key)
                        found = True
                    similar_list.append(val[1])
            overall_similar_list.append(similar_list)
            print(overall_similar_list)
            # check if the key has synonyms in text at all
            key_in_similar_list = any(
                key in sublist for sublist in overall_similar_list
            )
            # if key has no similarities, append the key itself
            if not found and not key_in_similar_list:
                new_ner_list.append(key)
            # otherwise, append an element from the similar words to the key
            elif found and key_in_similar_list:
                new_ner_list.append(similar_list[0])
        # last word is not in dict, so it needs to be appended separately if it is not already in the filtered list
        # it is already in the filtered list if there was a word before that is detected as synonym
        if len(list_to_filter) > 0:
            last_word_in_similar_list = any(
                list_to_filter[-1] in sublist for sublist in overall_similar_list
            )
            if not last_word_in_similar_list:
                new_ner_list.append(list_to_filter[-1])
        return new_ner_list

    @staticmethod
    def filter_out_synonyms(
        ner_list: List[str], biggest_allowed_distance: int
    ) -> List[str]:
        similar_dict = PreprocessArticles.calculate_word_distance(ner_list)
        new_ner_list = PreprocessArticles.build_list_without_synonyms(
            ner_list, similar_dict, biggest_allowed_distance
        )
        print(ner_list)
        print(new_ner_list)
        print(len(ner_list))
        print(len(new_ner_list))
        return new_ner_list

    def preprocessing(self, articles: pd.DataFrame):
        df_preprocessed_articles = articles.copy()
        df_preprocessed_articles = df_preprocessed_articles[:10]

        self.replace_new_line(df_preprocessed_articles)

        # remove special characters (regex)
        self.remove_special_characters(df_preprocessed_articles)

        # de_core_news_lg had the best score for entity recognition and syntax accuracy in german according to spacy.
        # for more information, see https://spacy.io/models/de#de_core_news_lg
        self.nlp = spacy.load("de_core_news_lg", disable=["parser"])

        # NER Tagging for persons and organizations
        df_preprocessed_articles = df_preprocessed_articles.apply(
            self.tag_dataframe, axis=1
        )

        # lowercase everything
        self.lowercase_article(df_preprocessed_articles)

        # stop word removal (after POS? -> filter unwanted POS)
        # evtl verbessern (opel fährt auf transporter auf --> opel fährt transporter)

        self.remove_stopwords(df_preprocessed_articles)

        # tokenization
        self.tokenization(df_preprocessed_articles)

        # POS tagging (before stemming? Could be used to count positive or negative adjectives etc.
        self.pos_tagging(df_preprocessed_articles)

        # stemming or lemmatization
        self.lemmatizing(df_preprocessed_articles)

        return df_preprocessed_articles
