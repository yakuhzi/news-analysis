from pathlib import Path

import pandas as pd

from src.preprocessing_articles import PreprocessArticles
from src.utils.reader import Reader
from src.utils.writer import Writer


def get_preprocessed_df(
    preprocessed_json_file: str, df_to_preprocess: pd.DataFrame
) -> pd.DataFrame:
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
    print(
        "Number of Tagesschau articles: {}".format(
            len(reader.df_tagesschau_articles.index)
        )
    )
    print("Number of TAZ articles: {}".format(len(reader.df_taz_articles.index)))

    print(reader.df_tagesschau_articles.dtypes)
    print(reader.df_tagesschau_articles.head())

    bild_preprocessed_file = "bild_preprocessed"
    df_bild_preprocessed = get_preprocessed_df(
        bild_preprocessed_file, reader.df_bild_articles
    )

    print(df_bild_preprocessed.dtypes)
    print(df_bild_preprocessed.head())

    tagesschau_preprocessed_file = "tagesschau_preprocessed"
    df_tagesschau_preprocessed = get_preprocessed_df(
        tagesschau_preprocessed_file, reader.df_tagesschau_articles
    )

    print(df_tagesschau_preprocessed.dtypes)
    print(df_tagesschau_preprocessed.head())

    taz_preprocessed_file = "taz_preprocessed"
    df_taz_preprocessed = get_preprocessed_df(
        taz_preprocessed_file, reader.df_taz_articles
    )

    print(df_taz_preprocessed.dtypes)
    print(df_taz_preprocessed.head())
