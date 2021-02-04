import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def hash(astring):
    return ord(astring[0])


class KMeansPolarity:
    def __init__(self, text: pd.Series):
        text = self._remove_special_characters(text)
        print(text)
        self.text = text.tolist()
        self.word_vectors = None

    def _remove_special_characters(self, text_series: pd.Series) -> pd.Series:
        text_series = text_series.str.join(" ")
        text_series = (
            text_series.str.replace(r"[^A-Za-zäöüÄÖÜß\-]", " ", regex=True)
            .str.replace(r" - ", " ", regex=False)
            .str.replace(r" +", " ", regex=True)
            .str.strip()
        )
        return text_series.str.split()

    def get_bigrams(self):
        phrases = Phrases(self.text, min_count=10)
        bigram = Phraser(phrases)

        return bigram[self.text]

    def get_word2vec(self, bigrams):
        w2v = Word2Vec(
            min_count=5,  # Ignores all words that have a total absolute frequency less than 2
            size=100,  # Dimensions of the embeddings: 100
            alpha=0.03,  # Initial learning rate of 0.03
            negative=20,  # 20 negative samples
            window=8,  # Window size 3
            min_alpha=0.0001,  # Make sure that the smallest learning rate does not go below 0.0001.
            sample=6e-5,
            # Set the threshold for configuring which higher-frequency words are randomly down-sampled to 6e-5
            hashfxn=hash,  # Set the hashfunction of the word2vec to the given function
            workers=1,  # Train on a single worker to make sure you get the same result as ours.
        )

        w2v.build_vocab(sentences=bigrams)
        w2v.train(sentences=bigrams, total_examples=len(bigrams), epochs=100)
        w2v.init_sims()
        return w2v.wv

    def k_means(self):
        model = KMeans(n_clusters=2, max_iter=1000, random_state=True, n_init=50).fit(X=self.word_vectors.vectors)
        # positive_cluster_center = model.cluster_centers_[0]
        # negative_cluster_center = model.cluster_centers_[1]
        similar_0 = self.word_vectors.similar_by_vector(model.cluster_centers_[0], topn=10, restrict_vocab=None)
        print(similar_0)
        similar_1 = self.word_vectors.similar_by_vector(model.cluster_centers_[1], topn=10, restrict_vocab=None)
        print(similar_1)

    def tsneplot(self, model, word):
        """Plot in seaborn the results from the t-SNE dimensionality reduction for the top 10 most similar and dissimilar words"""
        embs = np.empty((0, 100), dtype="f")  # to save all the embeddings
        word_labels = [word]
        color_list = ["green"]

        embs = np.append(embs, model.wv.__getitem__([word]), axis=0)

        # gets list of most similar words
        close_words = model.wv.most_similar([word])

        # gets list of most dissimilar words (get the sorted list of all the words and their similarity)
        all_sims = sorted(model.wv.most_similar([word], topn=len(model.wv.vocab)))
        # choose the bottom 10
        far_words = all_sims[:10]

        # adds the vector for each of the closest words to the array
        for wrd_score in close_words:
            wrd_vector = model.wv.__getitem__([wrd_score[0]])
            word_labels.append(wrd_score[0])
            color_list.append("blue")
            embs = np.append(embs, wrd_vector, axis=0)

        # adds the vector for each of the furthest words to the array
        for wrd_score in far_words:
            wrd_vector = model.wv.__getitem__([wrd_score[0]])
            word_labels.append(wrd_score[0])
            color_list.append("red")
            embs = np.append(embs, wrd_vector, axis=0)

        np.set_printoptions(suppress=True)
        Y = TSNE(n_components=2, random_state=42, perplexity=15).fit_transform(
            embs
        )  # with  n_components=2, random_state=42, perplexity=15

        # Sets everything up to plot
        df = pd.DataFrame(
            {"x": [x for x in Y[:, 0]], "y": [y for y in Y[:, 1]], "words": word_labels, "color": color_list}
        )

        fig, _ = plt.subplots()
        fig.set_size_inches(10, 10)

        # Basic plot
        p1 = sns.regplot(
            data=df, x="x", y="y", fit_reg=False, marker="o", scatter_kws={"s": 40, "facecolors": df["color"]}
        )

        # adds annotations one by one with a loop
        for line in range(0, df.shape[0]):
            p1.text(
                df["x"][line],
                df["y"][line],
                "  " + df["words"][line].title(),
                horizontalalignment="left",
                verticalalignment="bottom",
                size="medium",
                color=df["color"][line],
                weight="normal",
            ).set_size(15)

        plt.xlim(Y[:, 0].min() - 50, Y[:, 0].max() + 50)
        plt.ylim(Y[:, 1].min() - 50, Y[:, 1].max() + 50)

        plt.title("t-SNE visualization for {}".format(word.title()))
        plt.show()

    def calc_polarity(self):
        bigrams = self.get_bigrams()
        self.word_vectors = self.get_word2vec(bigrams)
        self.k_means()
        # test plot to see clustering
        self.tsneplot(self.word_vectors, "schlecht")
