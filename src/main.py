import pandas as pd

from preprocessing import Preprocessing
from sentiment_gui import SentimentGUI
from tfidf_sentiment import TfidfSentiment
from utils.arguments import parse_arguments
from utils.comparison import Comparison
from utils.labeling import Labeling
from utils.reader import Reader
from utils.writer import Writer

if __name__ == "__main__":
    args = parse_arguments()

    # Read articles from json
    df_articles = Reader.read_articles(args.number_of_articles)

    # Apply preprocessing
    preprocessing = Preprocessing()
    df_paragraphs = preprocessing.get_paragraphs(df_articles, overwrite=args.force_processing)

    # Calculate sentiment of paragraphs
    tfidf_sentiment = TfidfSentiment(df_paragraphs)
    tfidf_sentiment.calculate_sentiment_score()
    tfidf_sentiment.map_sentiment(overwrite=True)

    # Save paragraphs to disk
    if args.write:
        Writer.write_dataframe(df_paragraphs, "paragraphs")

    # Show GUI
    if args.show_gui:
        gui = SentimentGUI(df_paragraphs)
        gui.show_gui()

    # Label data
    if args.labeling:
        labeling = Labeling(df_paragraphs)
        labeling.label(start=0, end=50)

    # Compare labeled data with results
    if args.compare:
        comparison = Comparison("labeled_paragraphs")
        print("Optimal threshold: " + str(comparison.train_threshold()) + "\n")
        comparison.precision()
        comparison.recall()
        comparison.f1_score()
        comparison.accuracy()
