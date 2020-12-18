from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, HdpModel, LdaModel, LdaMulticore, LsiModel, TfidfModel
from gensim.models.wrappers import LdaMallet
from gensim.similarities import MatrixSimilarity
from pandas import DataFrame, Series


class TopicDetection:
    """
    Class that detects topics of a set of documents.

    Attributes:
    - text_data: Set of documents to detect the topic on.
    - dictionary: Gensim dictionary created from the text_data.
    - corpus: Gensim corpus of bag of words from the dictionary.
    """

    def __init__(self, text_data: List[List[str]]):
        self.text_data = text_data
        self.dictionary = Dictionary(text_data)
        self.corpus = [self.dictionary.doc2bow(text) for text in text_data]

    def get_tfidf(self):
        tfidf_model = TfidfModel(self.corpus)
        return tfidf_model[self.corpus]

    def get_document_similarity(self):
        tfidf = self.get_tfidf()
        index = MatrixSimilarity(tfidf, num_features=len(self.dictionary))
        # index = SparseMatrixSimilarity(tfidf, num_features=len(self.dictionary))
        similarity = index[self.corpus]
        return similarity

    def get_lsa_model(self, num_topics: int) -> LsiModel:
        lsi_model = LsiModel(self.corpus, num_topics=num_topics, id2word=self.dictionary)
        print(lsi_model.print_topics(num_words=4))
        return lsi_model

    def get_lda_model(self, num_topics: int) -> LdaModel:
        lda_model = LdaMulticore(self.corpus, num_topics=num_topics, id2word=self.dictionary)
        print(lda_model.print_topics(num_words=4))
        return lda_model

    def get_hdp_model(self) -> HdpModel:
        hdp_model = HdpModel(self.corpus, id2word=self.dictionary)
        print(hdp_model.print_topics(num_words=4))
        return hdp_model

    def visualize_topics(self, model: [LdaModel, HdpModel]) -> None:
        lda_display = pyLDAvis.gensim.prepare(model, self.corpus, self.dictionary)
        pyLDAvis.show(lda_display)

    def save_topics_per_document(self, lda_model: LdaModel, filename: str, head: int = 50) -> None:
        df_topics = self.__format_topics_sentences(model=lda_model)

        # Format
        df_dominant_topic = df_topics.reset_index()
        df_dominant_topic.columns = ["Document", "Dominant Topic", "Topic Contribution", "Keywords", "Text"]

        # Show
        markdown = df_dominant_topic.head(head).to_markdown()
        file = open(filename + ".md", "w", encoding="utf8")
        file.write(markdown)
        file.close()

    def calculate_coherence_score(self, lda_model: LdaModel) -> float:
        coherence_model = CoherenceModel(lda_model, texts=self.text_data, dictionary=self.dictionary, coherence="c_v")
        coherence = coherence_model.get_coherence()
        print("Coherence Score: ", coherence)
        return coherence

    def plot_coherence_scores(self, model: str, start: int = 5, limit: int = 40, step: int = 5) -> None:
        model_list, coherence_values = self.__compute_coherence_values(model, start=start, limit=limit, step=step)
        x = range(start, limit, step)

        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(model, loc="best")
        plt.show()

    def __compute_coherence_values(
        self, model: str, limit: int, start: int, step: int
    ) -> (List[LdaMallet], List[float]):
        model_list: List[LdaMallet] = []
        coherence_values: List[float] = []

        for num_topics in range(start, limit, step):
            print("Number of topics: ", num_topics)

            if model == "LSA":
                model = self.get_lsa_model(num_topics)
            else:
                model = self.get_lda_model(num_topics)

            model_list.append(model)

            coherence = self.calculate_coherence_score(model)
            coherence_values.append(coherence)

        return model_list, coherence_values

    def __format_topics_sentences(self, model: Union[LsiModel, LdaModel, HdpModel]) -> DataFrame:
        df_topics = DataFrame()

        # Get main topic in each document
        for i, row in enumerate(model[self.corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)

            # Get the Dominant topic, Contribution and Keywords for each document
            topic_num = row[0][0]
            topic_prob = row[0][1]

            wp = model.show_topic(topic_num)
            topic_keywords = ", ".join([word for word, prop in wp])

            series = Series([int(topic_num), round(topic_prob, 4), topic_keywords])
            df_topics = df_topics.append(series, ignore_index=True)

        df_topics.columns = ["Dominant Topic", "Contribution", "Topic Keywords"]

        # Add original text to the end of the output
        contents = Series(self.text_data)
        df_topics = pd.concat([df_topics, contents], axis=1)
        return df_topics
