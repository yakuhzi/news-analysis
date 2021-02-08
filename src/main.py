from topic_zero_shot import TopicZeroShot

from keyword_extraction import KeywordExtraction
from preprocessing import Preprocessing
from tfidf_sentiment import TfidfSentiment
from utils.reader import Reader
from utils.writer import Writer

if __name__ == "__main__":
    # Read articles from json
    df_articles = Reader.read_articles(10000)

    # Apply preprocessing
    preprocessing = Preprocessing()
    df_articles = preprocessing.get_articles(df_articles)
    # df_paragraphs = preprocessing.get_paragraphs(df_articles, overwrite=False)

    # Extract keywords with td-idf and show bipartite graph
    # keyword_extraction = KeywordExtraction(df_paragraphs)
    # df_term_weights = keyword_extraction.get_term_weight_tuples()

    topic_zero_shot = TopicZeroShot()

    samples = df_articles.sample(50)

    samples["topics_bart"] = topic_zero_shot.predict_topics(topic_zero_shot.bart_classifier, samples["title"])
    Writer.write(samples, "topic_articles")
    samples["topics_distilbart"] = topic_zero_shot.predict_topics(
        topic_zero_shot.distilbart_classifier, samples["title"]
    )
    Writer.write(samples, "topic_articles")
    samples["topics_roberta"] = topic_zero_shot.predict_topics(topic_zero_shot.roberta_classifier, samples["title"])
    Writer.write(samples, "topic_articles")

    # Calculate sentiment of paragraphs
    # tfidf_sentiment = TfidfSentiment(df_paragraphs)
    # tfidf_sentiment.add_sentiment()
    # tfidf_sentiment.map_sentiment()

    # Visualize sentiment by media and party
    # figures_by_party = Visualization.get_pie_charts(tfidf_sentiment.df_paragraphs, by_party=True)
    # figures_by_media = Visualization.get_pie_charts(tfidf_sentiment.df_paragraphs, by_party=False)
    # plt.show()

    # gui = SentimentGUI(df_paragraphs)
    # gui.show_gui()
