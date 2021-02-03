from pandas import DataFrame

from keyword_extraction import KeywordExtraction
from preprocessing import Preprocessing
from tfidf_sentiment import TfidfSentiment
from utils.reader import Reader
from utils.writer import Writer


def merge_paragrahps_with_titles() -> DataFrame:
    df_paragraphs_bild["title_nouns"] = df_paragraphs_bild["article_index"].apply(
        lambda index: df_titles_bild["nouns"][index]
    )

    df_paragraphs_tagesschau["title_nouns"] = df_paragraphs_tagesschau["article_index"].apply(
        lambda index: df_titles_tagesschau["nouns"][index]
    )

    df_paragraphs_taz["title_nouns"] = df_paragraphs_taz["article_index"].apply(
        lambda index: df_titles_taz["nouns"][index]
    )

    df_paragraphs = df_paragraphs_tagesschau.append(df_paragraphs_taz).append(df_paragraphs_bild)
    df_paragraphs["nouns"] = df_paragraphs["title_nouns"] + df_paragraphs["nouns"]
    return df_paragraphs


def extract_keywords() -> None:
    # Merge paragraphs with titles of articles to get more context
    df_paragraphs_and_titles = merge_paragrahps_with_titles()

    # Extract keywords with td-idf and shor bipartite graph
    keyword_extraction = KeywordExtraction(df_paragraphs_and_titles)
    df_term_weights = keyword_extraction.get_term_weight_tuples()
    keyword_extraction.show_graph(df_term_weights)


if __name__ == "__main__":
    reader = Reader()
    reader.read_articles(100)

    preprocessing = Preprocessing()
    df_paragraphs_bild, df_paragraphs_tagesschau, df_paragraphs_taz = preprocessing.get_paragraphs(reader)
    df_titles_bild, df_titles_tagesschau, df_titles_taz = preprocessing.get_titles(reader)

    df_paragraphs = df_paragraphs_tagesschau.append(df_paragraphs_taz).append(df_paragraphs_bild)
    tfidf_sentiment = TfidfSentiment(df_paragraphs)
    df_paragraphs = tfidf_sentiment.get_sentiment()

    Writer.write_articles(df_paragraphs, "paragraph_sentiments")
