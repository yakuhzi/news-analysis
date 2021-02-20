import re

from utils.writer import Writer


class Labeling:
    def __init__(self, dataframe):
        dataframe["original_index"] = dataframe.index
        self.dataframe = dataframe.sample(frac=1, random_state=0).reset_index(drop=True)

        parties = [
            "cdu",
            "union",
            "csu",
            "spd",
            "sozialdemokraten",
            "bündnis90 die grünen",
            "die grünen",
            "den grünen",
            "die grüne",
            "grünen",
            "grüne",
            "fdp",
            "liberalen",
            "freien demokrate",
            "freie demokratische partei",
            "afd",
            "alternative für deutschland",
            "den linken",
            "die linke",
            "linke",
        ]

        self.partyPattern = re.compile("|".join(parties), re.IGNORECASE)

    def label(self, start: int = 0, end: int = None):
        if end is None:
            end = len(self.dataframe)

        dataframe = self.dataframe.iloc[start:end]

        for index, row in dataframe.iterrows():
            title = row["title"]
            text = row["original_text"]
            persons = list(filter(lambda person: len(person) > 2, row["persons"]))

            if len(persons) > 0:
                personPattern = re.compile("|".join(persons), re.IGNORECASE)
                title = personPattern.sub("<Person>", title)
                text = personPattern.sub("<Person>", text)

            title = self.partyPattern.sub("<Partei>", title)
            text = self.partyPattern.sub("<Partei>", text)

            mediaPattern = re.compile("|".join(["bild", "bildplus", "taz", "tagesschau"]), re.IGNORECASE)
            title = mediaPattern.sub("<Zeitung>", title)
            text = mediaPattern.sub("<Zeitung>", text)

            print("================================================")
            print(title)
            print("++++++++++++++++++++++++++++++++++++++++++++++++")
            print(text)
            print("================================================")

            self.get_polarity_input(dataframe, index)
            self.get_subjectivity_input(dataframe, index)

            Writer.write_articles(dataframe, "labeled_paragraphs")

    def get_polarity_input(self, dataframe, index):
        while True:
            label = input("Polarity: ")

            if label == "0":
                dataframe.at[index, "labeled_sentiment"] = "Neutral"
                return
            if label == "1":
                dataframe.at[index, "labeled_sentiment"] = "Positive"
                return
            if label == "-1":
                dataframe.at[index, "labeled_sentiment"] = "Negative"
                return

    def get_subjectivity_input(self, dataframe, index):
        while True:
            label = input("Subjectivity: ")

            if label == "0":
                dataframe.at[index, "labeled_subjectivity"] = "Objective"
                return
            if label == "1":
                dataframe.at[index, "labeled_subjectivity"] = "Subjective"
                return
