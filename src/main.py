import sys
from pathlib import Path

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
    tfidf_sentiment.get_context_polarity(8)
    tfidf_sentiment.calculate_sentiment_score()
    tfidf_sentiment.map_sentiment()

    # Label data
    if args.labeling is not None:
        labeling = Labeling(df_paragraphs)
        start = args.labeling.split("-")[0]
        start = int(start) if start.isdigit() else 0
        end = args.labeling.split("-")[1]
        end = int(end) if end.isdigit() else 50
        labeling.label(start=start, end=end)

    # Train threshold for sentiment mapping
    if args.train:
        labeled_file = Path("src/output/labeled_paragraphs.json")

        if not labeled_file.exists():
            print('You have to provide a labeled file "labeled_paragraphs.json" for training in the output folder')
            sys.exit()

        comparison = Comparison(labeled_file)

        # Train the score threshold
        optimal_threshold = comparison.train_threshold()
        print("Optimal threshold: {}\n".format(optimal_threshold))

        # Train the window and the score threshold
        optimal_context_thresholds = comparison.train_context_thresholds()
        print(
            "Optimal context thresholds: {} (window), {} (score)\n".format(
                optimal_context_thresholds[0], optimal_context_thresholds[1]
            )
        )

    # Save paragraphs to disk
    Writer.write_dataframe(df_paragraphs, "paragraphs")

    # Show GUI
    if args.show_gui:
        gui = SentimentGUI(df_paragraphs)
        gui.show_gui()

    # Compare labeled data with results
    if args.compare:
        labeled_file = Path("src/output/labeled_paragraphs.json")

        if not labeled_file.exists():
            print('You have to provide a labeled file "labeled_paragraphs.json" for comparison in the output folder')
            sys.exit()

        comparison = Comparison(labeled_file)
        comparison.precision()
        comparison.recall()
        comparison.f1_score()
        comparison.accuracy()
