from preprocessing import Preprocessing
from sentiment_gui import SentimentGUI
from tfidf_sentiment import TfidfSentiment
from utils.reader import Reader
from utils.writer import Writer

if __name__ == "__main__":
    # Read articles from json
    df_articles = Reader.read_articles(1000)

    # Apply preprocessing
    preprocessing = Preprocessing()
    df_paragraphs = preprocessing.get_paragraphs(df_articles, overwrite=False)

    # Calculate sentiment of paragraphs
    tfidf_sentiment = TfidfSentiment(df_paragraphs)
    tfidf_sentiment.calculate_sentiment_score(overwrite=False)
    tfidf_sentiment.map_sentiment(overwrite=False)

    # Save paragraphs to disk
    Writer.write_dataframe(tfidf_sentiment.df_paragraphs, "paragraphs")

    # Show GUI
    gui = SentimentGUI(df_paragraphs)
    gui.show_gui()

    # Extract keywords with td-idf and show bipartite graph
    # keyword_extraction = KeywordExtraction(df_paragraphs)
    # df_weights_parties = keyword_extraction.get_term_weight_tuples(by_party=True, all_terms=True)
    # df_weights_media = keyword_extraction.get_term_weight_tuples(by_party=False, all_terms=True)
    # tripartite_graph = keyword_extraction.get_tripartite_graph(df_weights_parties, df_weights_media)
    # plt.show()

    # Visualize sentiment by media and party
    # figures_by_party = Visualization.get_pie_charts(tfidf_sentiment.df_paragraphs, by_party=True)
    # figures_by_media = Visualization.get_pie_charts(tfidf_sentiment.df_paragraphs, by_party=False)
    # plt.show()

    # Label data
    # labeling = Labeling(df_paragraphs)
    # labeling.label(start=0, end=50)

    # Compare labeled data with results
    # comparison = Comparison("labeled_paragraphs")
    # comparison.polarity()
    # comparison.polarity_to_subjectivity()
