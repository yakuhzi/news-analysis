from pathlib import Path

import pandas as pd
from ner_tagger import NERTagger

from src.preprocessing_articles import PreprocessArticles
from src.utils.reader import Reader
from src.utils.writer import Writer


def get_ner_df(ner_json_path: str, df_to_tag: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to get the ner tagged pandas dataframe. If the tagging already was done ones (JSON files with
    tags exist) the tagging is not done again but the json files with the tags are read into a pandas data frame.
    If ner tagging is proceeded, the result will be stored in a json file.

    Arguments:
    - ner_json_path: Path to json file to store/ read the results of ner tagging.
    - df_to_tag: data frame with the text to tag, if the data still needs to be tagged

    Return:
    - df_ner: Pandas data frame of the ner tagged input
    """
    if not Path(ner_json_path).exists():
        df_ner = apply_ner_tagging(df_to_tag)
        store_ner_tags_in_json(df_ner, ner_json_path)
    else:
        df_ner = reader.read_json_to_df_default(ner_json_path)
    return df_ner


def apply_ner_tagging(df_to_tag: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the ner tagging on a given data frame

    Arguments:
    - df_to_tag: the Pandas data frame which should be tagged

    Return:
    - df_ner: Pandas data frame of the ner tagged input
    """
    ner_tagger = NERTagger()
    df_ner = df_to_tag.apply(ner_tagger.tag_dataframe, axis=1)
    df_ner = df_ner.filter(["persons", "organizations"], axis=1)
    return df_ner


def store_ner_tags_in_json(df_ner: pd.DataFrame, path: str) -> None:
    """
    Helper function to store Pandas dataframe of ner tagging into json file

    Arguments:
    - df_ner: the Pandas data frame which should be stored in json
    - path: the path where the dataframe should be stored
    """
    with open(path, "w", encoding="utf-8") as file:
        df_ner.to_json(file, force_ascii=False, orient="records")


if __name__ == "__main__":
    reader = Reader()
    reader.read_articles()

    print("Number of Bild articles: {}".format(len(reader.df_bild_articles.index)))
    print(
        "Number of Tagesschau articles: {}".format(
            len(reader.df_tagesschau_articles.index)
        )
    )
    print("Number of TAZ articles: {}".format(len(reader.df_taz_articles.index)))

    print(reader.df_tagesschau_articles.dtypes)
    print(reader.df_tagesschau_articles.head())

    preprocess = PreprocessArticles()
    bild = preprocess.preprocessing(reader.df_bild_articles)
    writer = Writer()
    writer.write_articles(bild, "bild_preprocessed")

    bild_ner_path = "src/data/bild_ner.json"
    df_bild_ner = get_ner_df(bild_ner_path, reader.df_bild_articles)

    print(df_bild_ner.dtypes)
    print(df_bild_ner.head())

    tagesschau_ner_path = "src/data/tagesschau_ner.json"
    df_tagesschau_ner = get_ner_df(tagesschau_ner_path, reader.df_tagesschau_articles)

    print(df_tagesschau_ner.dtypes)
    print(df_tagesschau_ner.head())

    taz_ner_path = "src/data/taz_ner.json"
    df_taz_ner = get_ner_df(taz_ner_path, reader.df_taz_articles)

    print(df_taz_ner.dtypes)
    print(df_taz_ner.head())
