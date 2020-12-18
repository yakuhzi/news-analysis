from pathlib import Path
from typing import List

import regex as re
from pandas import DataFrame
from topic_detection import TopicDetection

from src.preprocessing_articles import PreprocessArticles
from src.utils.reader import Reader
from src.utils.writer import Writer


def get_preprocessed_df(preprocessed_json_file: str, df_to_preprocess: DataFrame) -> DataFrame:
    """
    Helper function to get the preprocessed pandas dataframe. If the preprocessing already was done ones (JSON files
    exist) the tagging is not done again but the json files with the perprocessing are read into a pandas data frame.
    If preprocessing is proceeded, the result will be stored in a json file.

    Arguments:
    - preprocessed_json_file: Name of json file to store/ read the results of preprocessing.
    - df_to_preprocess: data frame with the text to preprocess, if the data still needs to be preprocessed

    Return:
    - df_preprocessed: Pandas data frame of the preprocessed input
    """
    preprocessed_json_path = "src/data/" + preprocessed_json_file + ".json"

    if not Path(preprocessed_json_path).exists():
        preprocess = PreprocessArticles()
        df_preprocessed = preprocess.preprocessing(df_to_preprocess)
        Writer.write_articles(df_preprocessed, preprocessed_json_file)
    else:
        df_preprocessed = reader.read_json_to_df_default(preprocessed_json_path)
    return df_preprocessed


if __name__ == "__main__":
    reader = Reader()
    reader.read_articles()

    print("Number of Bild articles: {}".format(len(reader.df_bild_articles.index)))
    print("Number of Tagesschau articles: {}".format(len(reader.df_tagesschau_articles.index)))
    print("Number of TAZ articles: {}".format(len(reader.df_taz_articles.index)))

    bild_preprocessed_file = "bild_preprocessed"
    df_bild_preprocessed = get_preprocessed_df(bild_preprocessed_file, reader.df_bild_articles)

    print(df_bild_preprocessed.dtypes)
    print(df_bild_preprocessed.head())

    tagesschau_preprocessed_file = "tagesschau_preprocessed"
    df_tagesschau_preprocessed = get_preprocessed_df(tagesschau_preprocessed_file, reader.df_tagesschau_articles)

    print(df_tagesschau_preprocessed.dtypes)
    print(df_tagesschau_preprocessed.head())

    taz_preprocessed_file = "taz_preprocessed"
    df_taz_preprocessed = get_preprocessed_df(taz_preprocessed_file, reader.df_taz_articles)

    print(df_taz_preprocessed.dtypes)
    print(df_taz_preprocessed.head())

    df_articles = df_bild_preprocessed.append(df_tagesschau_preprocessed).append(df_taz_preprocessed)
    print(len(df_articles))

    text_data: List[List[str]] = []

    for index, row in df_articles.sample(n=1000).iterrows():
        text = re.sub(" \\d+", "", row["text"])
        text_data.append(text.split(" "))

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

    # topic_detection.plot_coherence_scores("LSA", start=5, limit=50, step=5)
    # topic_detection.plot_coherence_scores("LDA", start=5, limit=50, step=5)

    topic_detection.visualize_topics(lda_model)
