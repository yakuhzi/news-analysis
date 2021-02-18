from utils.writer import Writer


class Labeling:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def label(self, start: int = 0, end: int = None):
        if end is None:
            end = len(self.dataframe)

        dataframe = self.dataframe.iloc[start:end]
        print(dataframe)

        for index, row in dataframe.iterrows():
            print("================================================")
            print(row["title"])
            print("++++++++++++++++++++++++++++++++++++++++++++++++")
            print(row["original_text"])
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
