import time
from typing import List

from transformers import pipeline


class TopicZeroShot:
    def __init__(self, text_data: List[List[str]]):
        self.text_data = text_data
        # self.classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
        self.classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-9")
        self.hypothesis_template = "This text is about {}."

    def predict(self, documents, topics):
        sequence = list(map(lambda document: " ".join(document), documents))
        print(sequence)

        start = time.time()

        result = self.classifier(sequence, topics, hypothesis_template=self.hypothesis_template, multi_class=True)
        print(result)

        end = time.time()
        print(end - start)
