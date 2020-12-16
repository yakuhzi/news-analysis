from typing import List

import spacy
from spacy.lang.de import German
from utils.reader import Reader
from utils.topic_detection import TopicDetection

if __name__ == "__main__":
    reader = Reader()
    reader.read_articles()

    print("Number of Bild articles: {}".format(len(reader.df_bild_articles.index)))
    print("Number of Tagesschau articles: {}".format(len(reader.df_tagesschau_articles.index)))
    print("Number of TAZ articles: {}".format(len(reader.df_taz_articles.index)))

    print(reader.df_tagesschau_articles.dtypes)
    print(reader.df_tagesschau_articles.head())

    articles = reader.df_tagesschau_articles.head(100)

    nlp = spacy.load("de")
    nlp.disable_pipes("ner")

    stopwords = spacy.lang.de.stop_words.STOP_WORDS

    # nltk.download("wordnet")
    # nltk.download("stopwords")

    text_data: List[List[str]] = []

    for index, row in articles.iterrows():
        doc = nlp(row["text"])
        tokens = [token for token in doc if len(token) > 4]
        tokens = [token for token in tokens if token not in stopwords]
        tokens = [token for token in tokens if token.pos_ == "NOUN"]
        tokens = list(map(lambda token: token.lemma_, tokens))
        text_data.append(tokens)

    topic_detection = TopicDetection(text_data)
    lda_model = topic_detection.get_lda_model()
    topic_detection.calculate_coherence_score(lda_model)
    # topic_detection.plot_coherence_scores(limit=40, step=10)
    # topic_detection.save_topics_per_document(lda_model)
