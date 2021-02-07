from keyword_extraction import KeywordExtraction
from preprocessing import Preprocessing
from tfidf_sentiment import TfidfSentiment
from utils.reader import Reader
from utils.visualization import Visualization

if __name__ == "__main__":
    # Read articles from json
    df_articles = Reader.read_articles(10000)

    # Apply preprocessing
    preprocessing = Preprocessing()
    df_paragraphs = preprocessing.get_paragraphs(df_articles, overwrite=False)

    # Extract keywords with td-idf and show bipartite graph
    keyword_extraction = KeywordExtraction(df_paragraphs)
    df_term_weights = keyword_extraction.get_term_weight_tuples()
    keyword_extraction.show_graph(df_term_weights)

    # Calculate sentiment of paragraphs
    tfidf_sentiment = TfidfSentiment(df_paragraphs)
    tfidf_sentiment.add_sentiment()
    tfidf_sentiment.map_sentiment()

    # Visualize sentiment by media and party
    Visualization.show_pie_charts(tfidf_sentiment.df_paragraphs, by_party=True)
    Visualization.show_pie_charts(tfidf_sentiment.df_paragraphs, by_party=False)
