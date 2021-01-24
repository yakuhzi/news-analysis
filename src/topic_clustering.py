from typing import List

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


class TopicClustering:
    def __init__(self, text_data: List[List[str]]):
        self.text_data = text_data

    def train(self):
        document = list(map(lambda text: " ".join(text), self.text_data))

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(document)

        true_k = 20
        model = KMeans(n_clusters=true_k, init="k-means++", max_iter=100, n_init=1)
        model.fit(X)

        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()

        for i in range(true_k):
            print("Cluster %d:" % i),

            for ind in order_centroids[i, :10]:
                print(" %s" % terms[ind])

        return model, vectorizer

    def predict(self, model, vectorizer, document):
        document = " ".join(document)
        print(document)

        print("Prediction")
        X = vectorizer.transform([document])

        predicted = model.predict(X)
        print(predicted)
