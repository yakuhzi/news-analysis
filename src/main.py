from preprocessing import Preprocessing
from tfidf_sentiment import TfidfSentiment
from utils.comparison import Comparison
from utils.labeling import Labeling
from utils.reader import Reader

if __name__ == "__main__":
    # Read articles from json
    df_articles = Reader.read_articles(50)

    # Apply preprocessing
    preprocessing = Preprocessing()
    df_paragraphs = preprocessing.get_paragraphs(df_articles, overwrite=False)

    # Extract keywords with td-idf and show bipartite graph
    # keyword_extraction = KeywordExtraction(df_paragraphs)
    # df_term_weights = keyword_extraction.get_term_weight_tuples()
    # bipartite_graph = keyword_extraction.get_graph(df_term_weights)
    # plt.show()

    # Calculate sentiment of paragraphs
    tfidf_sentiment = TfidfSentiment(df_paragraphs)
    tfidf_sentiment.add_sentiment(overwrite=False)
    tfidf_sentiment.map_sentiment(overwrite=True)

    # Label data
    labeling = Labeling(df_paragraphs)
    labeling.label(start=0, end=50)

    # Compare labeled data with results
    comparison = Comparison("labeled_paragraphs")
    comparison.polarity()
    comparison.polarity_to_subjectivity()

    # Visualize sentiment by media and party
    # figures_by_party = Visualization.get_pie_charts(tfidf_sentiment.df_paragraphs, by_party=True)
    # figures_by_media = Visualization.get_pie_charts(tfidf_sentiment.df_paragraphs, by_party=False)
    # plt.show()

    # gui = SentimentGUI(df_paragraphs)
    # gui.show_gui()
