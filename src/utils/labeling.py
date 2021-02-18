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
            "die grüne",
            "die grünen",
            "den grünen",
            "grünen",
            "grüne",
            "fdp",
            "liberalen",
            "freien demokrate",
            "freie demokratische partei",
            "afd",
            "alternative für deutschland",
            "den linken" "die linke",
            "linke",
        ]

        persons = [item for sublist in dataframe["persons"].tolist() for item in sublist]
        self.partyPattern = re.compile("|".join(parties), re.IGNORECASE)
        self.personPattern = re.compile("|".join(persons), re.IGNORECASE)

    def label(self, start: int = 0, end: int = None):
        if end is None:
            end = len(self.dataframe)

        dataframe = self.dataframe.iloc[start:end]

        for index, row in dataframe.iterrows():
            title = self.partyPattern.sub("<Partei>", row["title"])
            title = self.personPattern.sub("<Person>", title)

            text = self.partyPattern.sub("<Partei>", row["original_text"])
            text = self.personPattern.sub("<Person>", text)

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
