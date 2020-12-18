from typing import List

import spacy
from spacy.lang.de import German
from topic_detection import TopicDetection
from utils.reader import Reader

if __name__ == "__main__":
    reader = Reader()
    reader.read_articles()

    print("Number of Bild articles: {}".format(len(reader.df_bild_articles.index)))
    print("Number of Tagesschau articles: {}".format(len(reader.df_tagesschau_articles.index)))
    print("Number of TAZ articles: {}".format(len(reader.df_taz_articles.index)))

    print(reader.df_tagesschau_articles.dtypes)
    print(reader.df_tagesschau_articles.head())

    articles = reader.df_tagesschau_articles.append(reader.df_bild_articles).append(reader.df_taz_articles)
    political_articles = articles[articles["text"].str.contains(r"CDU|FDP|AfD|Grüne|SPD|Linke")]
    print(len(political_articles))

    nlp = spacy.load("de")
    nlp.disable_pipes("ner")

    stopwords = spacy.lang.de.stop_words.STOP_WORDS
    stopwords |= {"bildplus", "mensch", "jahr", "deutschland", "taz", "rtr"}

    # nltk.download("wordnet")
    # nltk.download("stopwords")

    text_data: List[List[str]] = []

    for index, row in political_articles.sample(n=200).iterrows():
        doc = nlp(row["text"])
        tokens = [token for token in doc if len(token) > 2]
        tokens = [token for token in tokens if token.pos_ == "NOUN"]
        tokens = list(map(lambda token: token.lemma_, tokens))
        tokens = [token for token in tokens if token.lower() not in stopwords]
        text_data.append(tokens)

    topic_detection = TopicDetection(text_data)

    lsa_model = topic_detection.get_lsa_model(num_topics=20)
    lda_model = topic_detection.get_lda_model(num_topics=20)
    hdp_model = topic_detection.get_hdp_model()

    topic_detection.calculate_coherence_score(lsa_model)
    topic_detection.calculate_coherence_score(lda_model)
    topic_detection.calculate_coherence_score(hdp_model)

    topic_detection.save_topics_per_document(lsa_model, "src/data/lsa_topics")
    topic_detection.save_topics_per_document(lda_model, "src/data/lda_topics")
    topic_detection.save_topics_per_document(hdp_model, "src/data/hdp_topics")

    topic_detection.plot_coherence_scores("LSA", start=5, limit=200, step=40)
    topic_detection.plot_coherence_scores("LDA", start=5, limit=200, step=40)

    # topic_detection.visualize_topics(hdp_model)
