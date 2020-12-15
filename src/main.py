from src.preprocessing_Articles import Preprocess_Articles
from src.utils.reader import Reader

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

    preprocess = Preprocess_Articles()
    preprocess.preprocessing(reader.df_bild_articles)
