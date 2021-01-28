import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from src.preprocessing import Preprocessing
from src.utils.document_type import DocumentType
from src.utils.reader import Reader
from src.utils.writer import Writer


def get_preprocessed_df(
    preprocessed_json_file: str, df_to_preprocess: DataFrame, document_type: DocumentType, set_article_index=False
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
        preprocess = Preprocessing()
        df_preprocessed = preprocess.preprocessing(df_to_preprocess, document_type)
        Writer.write_articles(df_preprocessed, preprocessed_json_file)
    else:
        df_preprocessed = reader.read_json_to_df_default(preprocessed_json_path, set_article_index)
    return df_preprocessed


def filter_nouns(row: List[List[str]]) -> str:
    noun_pairs = filter(lambda word_pair: word_pair[1] == "NN", row)
    nouns = map(lambda word_pair: word_pair[0], noun_pairs)
    return " ".join(list(nouns))


def preprocess_articles():
    # Preprocess articles
    bild_preprocessed_file = "bild_preprocessed"
    df_bild_preprocessed = get_preprocessed_df(
        bild_preprocessed_file, reader.df_bild_articles, DocumentType.ARTICLE, set_article_index=True
    )

    print(df_bild_preprocessed.dtypes)
    print(df_bild_preprocessed.head())

    tagesschau_preprocessed_file = "tagesschau_preprocessed"
    df_tagesschau_preprocessed = get_preprocessed_df(
        tagesschau_preprocessed_file, reader.df_tagesschau_articles, DocumentType.ARTICLE, set_article_index=True
    )

    print(df_tagesschau_preprocessed.dtypes)
    print(df_tagesschau_preprocessed.head())

    taz_preprocessed_file = "taz_preprocessed"
    df_taz_preprocessed = get_preprocessed_df(
        taz_preprocessed_file, reader.df_taz_articles, DocumentType.ARTICLE, set_article_index=True
    )

    print(df_taz_preprocessed.dtypes)
    print(df_taz_preprocessed.head())

    return df_tagesschau_preprocessed.append(df_taz_preprocessed).append(df_bild_preprocessed)


def preprocess_titles():
    bild_preprocessed_titles_file = "bild_preprocessed_titles"
    df_bild_preprocessed_titles = get_preprocessed_df(
        bild_preprocessed_titles_file, reader.df_bild_articles, DocumentType.TITLE
    )

    print(df_bild_preprocessed_titles.dtypes)
    print(df_bild_preprocessed_titles.head())

    tagesschau_preprocessed_titles_file = "tagesschau_preprocessed_titles"
    df_tagesschau_preprocessed_titles = get_preprocessed_df(
        tagesschau_preprocessed_titles_file, reader.df_tagesschau_articles, DocumentType.TITLE
    )

    print(df_tagesschau_preprocessed_titles.dtypes)
    print(df_tagesschau_preprocessed_titles.head())

    taz_preprocessed_titles_file = "taz_preprocessed_titles"
    df_taz_preprocessed_titles = get_preprocessed_df(
        taz_preprocessed_titles_file, reader.df_taz_articles, DocumentType.TITLE
    )

    print(df_taz_preprocessed_titles.dtypes)
    print(df_taz_preprocessed_titles.head())

    return df_tagesschau_preprocessed_titles.append(df_taz_preprocessed_titles).append(df_bild_preprocessed_titles)


def preprocess_paragraphs():
    # Preprocess paragraphs of articles
    bild_preprocessed_paragraphs_file = "bild_preprocessed_paragraphs"
    df_bild_preprocessed_paragraphs = get_preprocessed_df(
        bild_preprocessed_paragraphs_file,
        reader.df_bild_articles,
        DocumentType.PARAGRAPH,
    )

    print(df_bild_preprocessed_paragraphs.dtypes)
    print(df_bild_preprocessed_paragraphs.head())

    tagesschau_preprocessed_paragraphs_file = "tagesschau_preprocessed_paragraphs"
    df_tagesschau_preprocessed_paragraphs = get_preprocessed_df(
        tagesschau_preprocessed_paragraphs_file, reader.df_tagesschau_articles, DocumentType.PARAGRAPH
    )

    print(df_tagesschau_preprocessed_paragraphs.dtypes)
    print(df_tagesschau_preprocessed_paragraphs.head())

    taz_preprocessed_paragraphs_file = "taz_preprocessed_paragraphs"
    df_taz_preprocessed_paragraphs = get_preprocessed_df(
        taz_preprocessed_paragraphs_file, reader.df_taz_articles, DocumentType.PARAGRAPH
    )

    print(df_taz_preprocessed_paragraphs.dtypes)
    print(df_taz_preprocessed_paragraphs.head())

    print("------------------------")
    # concat_df = pd.concat([df_bild_preprocessed_paragraphs[['article_index', 'nouns']].set_index('article_index'), all_articles_df["title"]],
    # axis=1, join='inner')

    df_bild_preprocessed_paragraphs["title"] = df_bild_preprocessed_paragraphs["article_index"].apply(
        lambda index: reader.df_bild_articles["title"][index]
    )
    print(df_bild_preprocessed_paragraphs.head())

    return df_tagesschau_preprocessed_paragraphs.append(df_taz_preprocessed_paragraphs).append(
        df_bild_preprocessed_paragraphs
    )


def get_party_paragraphs(party: str):
    return df_paragraphs[df_paragraphs.apply(lambda row: len(row["parties"]) == 1 and party in row["parties"], axis=1)]


def sort_out_position(party_string):
    # ex = "Ministerpräsident hatte gesagt, dass er heute Bundeswirtschaftsminister mit Kanzlerin trifft".lower().split()
    B = [
        "minister",
        "kanzler",
        "bürgermeister",
        "montag",
        "dienstag",
        "mittwoch",
        "donnerstag",
        "freitag",
        "samstag",
        "sonntag",
        "cdu",
        "csu",
        "spd",
        "grüne",
        "linke",
        "afd",
        "fdp",
        "partei",
        "sprecher",
        "bundesregierung",
    ]
    # print(ex)
    blacklist = re.compile("|".join([re.escape(word) for word in B]))
    processed = " ".join([word for word in party_string.lower().split() if not blacklist.search(word)])
    print(processed)
    return processed


def remove_party_positions(party_df):
    party_df["nouns_title"] = party_df["nouns"] + party_df["title"]
    print(party_df["nouns_title"])
    party_df["nouns"] = party_df["nouns"].apply(lambda words: sort_out_position(words))
    return party_df
    # ex = "Ministerpräsident hatte gesagt, dass er heute Bundeswirtschaftsminister mit Kanzlerin trifft".lower().split()
    # B  = ["minister", "kanzler"]
    # blacklist = re.compile('|'.join([re.escape(word) for word in B]))
    # print(" ".join([word for word in ex if not blacklist.search(word)]))


if __name__ == "__main__":
    reader = Reader()
    reader.read_articles(13000)

    print("Number of Bild articles: {}".format(len(reader.df_bild_articles.index)))
    print("Number of Tagesschau articles: {}".format(len(reader.df_tagesschau_articles.index)))
    print("Number of TAZ articles: {}".format(len(reader.df_taz_articles.index)))

    all_articles_df = reader.df_tagesschau_articles.append(reader.df_taz_articles).append(reader.df_bild_articles)
    print(all_articles_df)
    df_paragraphs = preprocess_paragraphs()
    # print(len(df_paragraphs))

    df_paragraphs["nouns"] = df_paragraphs["pos_tags"].apply(lambda row: filter_nouns(row))
    # text_data: List[List[str]] = df_paragraphs["nouns"].tolist()

    # print(df_paragraphs["nouns"])
    # text_data: List[List[str]] = df_paragraphs["text"].apply(lambda row: row.split()).tolist()
    # text_data = list(filter(lambda text: len(text) > 10, text_data))

    text_data_cdu = get_party_paragraphs("CDU")
    text_data_csu = get_party_paragraphs("CSU")
    text_data_spd = get_party_paragraphs("SPD")
    text_data_fdp = get_party_paragraphs("FDP")
    text_data_afd = get_party_paragraphs("AfD")
    text_data_gruene = get_party_paragraphs("Grüne")
    text_data_linke = get_party_paragraphs("Linke")

    # print(text_data_cdu)

    # Vectorize character_words
    vectorizer = CountVectorizer(min_df=5)
    count_vectorized = vectorizer.fit_transform(df_paragraphs["nouns"])

    # Apply Tfidf to count_vectorized
    transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    # transformed_words = transformer.fit_transform(count_vectorized)
    # weights = np.asarray(transformed_words.mean(axis=0)).ravel().tolist()
    # weights_df = DataFrame({"term": vectorizer.get_feature_names(), "TF_IDF": weights})
    # weights_df = weights_df.sort_values("TF_IDF", ascending=False).reset_index(drop=False)[:10]
    # weights_df.plot.bar(x="term", y="TF_IDF", rot=15, width=0.7, figsize=(10, 5))

    # you only needs to do this once, this is a mapping of index to
    # feature_names = vectorizer.get_feature_names()

    # get the document that we want to extract keywords from
    text_data_linke = remove_party_positions(text_data_linke)
    doc = text_data_linke["nouns"].to_string()

    # print(doc)

    # generate tf-idf for the given document
    transformer.fit(count_vectorized)
    tf_idf_vector = transformer.transform(vectorizer.transform([doc]))

    weights = np.asarray(tf_idf_vector.mean(axis=0)).ravel().tolist()
    weights_df = DataFrame({"term": vectorizer.get_feature_names(), "TF_IDF": weights})
    weights_df = weights_df.sort_values("TF_IDF", ascending=False).reset_index(drop=False)[:20]
    weights_df.plot.bar(x="term", y="TF_IDF", rot=15, width=0.7, figsize=(10, 5))

    # now print the results
    # print("\n=====Doc=====")
    # print(doc)
    # print("\n===Keywords===")
    # for k in keywords:
    #     print(k, keywords[k])

    plt.title("TF_IDF", size=15)
    plt.show()
