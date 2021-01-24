from pathlib import Path
from typing import List

from pandas import DataFrame
from topic_clustering import TopicClustering
from topic_zero_shot import TopicZeroShot

from src.preprocessing_articles import PreprocessArticles
from src.utils.reader import Reader
from src.utils.writer import Writer


def get_preprocessed_df(
    preprocessed_json_file: str, df_to_preprocess: DataFrame, split_paragraphs=True, set_article_index=False
) -> DataFrame:
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
        df_preprocessed = preprocess.preprocessing(df_to_preprocess, split_paragraphs)
        Writer.write_articles(df_preprocessed, preprocessed_json_file)
    else:
        df_preprocessed = reader.read_json_to_df_default(preprocessed_json_path, set_article_index)
    return df_preprocessed


def filter_nouns(row: List[List[str]]) -> List[str]:
    noun_pairs = filter(lambda word_pair: word_pair[1] == "NN", row)
    nouns = map(lambda word_pair: word_pair[0], noun_pairs)
    return list(nouns)


def preprocess_articles(reader: Reader):
    # Preprocess articles
    bild_preprocessed_file = "bild_preprocessed"
    df_bild_preprocessed = get_preprocessed_df(
        bild_preprocessed_file, reader.df_bild_articles, split_paragraphs=False, set_article_index=True
    )

    print(df_bild_preprocessed.dtypes)
    print(df_bild_preprocessed.head())

    tagesschau_preprocessed_file = "tagesschau_preprocessed"
    df_tagesschau_preprocessed = get_preprocessed_df(
        tagesschau_preprocessed_file, reader.df_tagesschau_articles, split_paragraphs=False, set_article_index=True
    )

    print(df_tagesschau_preprocessed.dtypes)
    print(df_tagesschau_preprocessed.head())

    taz_preprocessed_file = "taz_preprocessed"
    df_taz_preprocessed = get_preprocessed_df(
        taz_preprocessed_file, reader.df_taz_articles, split_paragraphs=False, set_article_index=True
    )

    print(df_taz_preprocessed.dtypes)
    print(df_taz_preprocessed.head())

    return df_tagesschau_preprocessed.append(df_taz_preprocessed).append(df_bild_preprocessed)


def preprocess_paragraphs(reader: Reader):
    # Preprocess paragraphs of articles
    bild_preprocessed_paragraphs_file = "bild_preprocessed_paragraphs"
    df_bild_preprocessed_paragraphs = get_preprocessed_df(bild_preprocessed_paragraphs_file, reader.df_bild_articles)

    print(df_bild_preprocessed_paragraphs.dtypes)
    print(df_bild_preprocessed_paragraphs.head())

    tagesschau_preprocessed_paragraphs_file = "tagesschau_preprocessed_paragraphs"
    df_tagesschau_preprocessed_paragraphs = get_preprocessed_df(
        tagesschau_preprocessed_paragraphs_file, reader.df_tagesschau_articles
    )

    print(df_tagesschau_preprocessed_paragraphs.dtypes)
    print(df_tagesschau_preprocessed_paragraphs.head())

    taz_preprocessed_paragraphs_file = "taz_preprocessed_paragraphs"
    df_taz_preprocessed_paragraphs = get_preprocessed_df(taz_preprocessed_paragraphs_file, reader.df_taz_articles)

    print(df_taz_preprocessed_paragraphs.dtypes)
    print(df_taz_preprocessed_paragraphs.head())

    return df_tagesschau_preprocessed_paragraphs.append(df_taz_preprocessed_paragraphs).append(
        df_bild_preprocessed_paragraphs
    )


if __name__ == "__main__":
    reader = Reader()
    reader.read_articles(1000)

    print("Number of Bild articles: {}".format(len(reader.df_bild_articles.index)))
    print("Number of Tagesschau articles: {}".format(len(reader.df_tagesschau_articles.index)))
    print("Number of TAZ articles: {}".format(len(reader.df_taz_articles.index)))

    df_paragraphs = preprocess_paragraphs(reader)
    print(len(df_paragraphs))

    df_paragraphs["nouns"] = df_paragraphs["pos_tags"].apply(lambda row: filter_nouns(row))
    text_data: List[List[str]] = df_paragraphs["nouns"].tolist()

    print(text_data)

    topics = [
        "Innenpolitik",
        "Außenpolitik",
        "Wirtschaft",
        "Finanzen",
        "Umwelt",
        "Wissenschaft",
        "Justiz",
        "Corona",
        "Arbeit & Soziales",
        "Landwirtschaft",
        "Sicherheit",
    ]

    topic_zero_shot = TopicZeroShot(text_data[:-1])
    topic_zero_shot.predict(text_data[:1], topics)

    # topic_clustering = TopicClustering(text_data[:-1])
    # model, vectorizer = topic_clustering.train()
    # topic_clustering.predict(model, vectorizer, text_data[-1])

    # for index, row in df_articles.sample(n=5).iterrows():
    #     text = re.sub(" \\d+", "", row["text"])
    #     text_data.append(text.split(" "))

    # topic_classification = TopicClassification(text_data[:-1])
    # llda_model = topic_classification.train_llda()
    #
    # document = " ".join(text_data[-1])
    # document = "example llda model example example good perfect good perfect good perfect" * 100
    # print(document)
    # topics = topic_classification.predict_llda(llda_model, document)
    # print(topics)
    #
    # llda_model.save_model_to_dir("./")
    # print(llda_model.top_terms_of_topic("Social Affairs and Labour Market", 15, True))
    # print(llda_model.top_terms_of_topic("Culture and Education", 15, True))
    # print(llda_model.top_terms_of_topic("Agriculture", 15, True))
    # print(llda_model.top_terms_of_topic("Finance", 15, True))
    # print(llda_model.top_terms_of_topic("Justice", 15, True))
    # print(llda_model.top_terms_of_topic("Internal Affairs", 15, True))
    # print(llda_model.top_terms_of_topic("Environment and Regional Planning", 15, True))
    # print(llda_model.top_terms_of_topic("Security and Foreign Affairs", 15, True))

    # topic_detection = TopicDetection(text_data)
    #
    # lsa_model = topic_detection.get_lsa_model(num_topics=20)
    # lda_model = topic_detection.get_lda_model(num_topics=20)
    # hdp_model = topic_detection.get_hdp_model()

    # topic_detection.calculate_coherence_score(lsa_model)
    # topic_detection.calculate_coherence_score(lda_model)
    # topic_detection.calculate_coherence_score(hdp_model)

    # topic_detection.save_topics_per_document(lsa_model, "src/data/lsa_topics")
    # topic_detection.save_topics_per_document(lda_model, "src/data/lda_topics")
    # topic_detection.save_topics_per_document(hdp_model, "src/data/hdp_topics")
    #
    # # topic_detection.plot_coherence_scores("LSA", start=5, limit=50, step=5)
    # # topic_detection.plot_coherence_scores("LDA", start=5, limit=50, step=5)
    #
    # topic_detection.visualize_topics(lda_model)
