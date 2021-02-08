import time
from typing import List

from pandas import DataFrame, Series
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
from transformers import pipeline


class TopicZeroShot:
    def __init__(self):
        tokenizer = Tokenizer(WordLevel())
        tokenizer.pre_tokenizer = Whitespace()

        self.bart_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.distilbart_classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-9")
        self.roberta_classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")
        self.hypothesis_template = "Dieser Text handelt von {}."

        self.topics = [
            "Innenpolitik",
            "Außenpolitik",
            "Wirtschaft & Finanzen",
            "Wissenschaft & Umwelt",
            "Corona",
            "Arbeit & Soziales",
            "Landwirtschaft",
            "Sicherheit & Justiz",
            "Bildung",
            "Wahlkampf & Umfragen",
        ]

    def predict_topics(self, classifier, word_series: Series):
        tqdm.pandas(desc="Predict topics")
        return word_series.progress_apply(lambda nouns: self._get_topics(classifier, nouns))

    def _get_topics(self, classifier, nouns: List[str]) -> List[str]:
        if len(nouns) == 0:
            return []

        # sequence = " ".join(nouns)

        result = classifier(nouns, self.topics, hypothesis_template=self.hypothesis_template, multi_class=True)
        topic_scores = zip(result["labels"], result["scores"])
        top = list(filter(lambda x: x[1] > 0.5, topic_scores))
        return list(map(lambda x: x[0], top))
