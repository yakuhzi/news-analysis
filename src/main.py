from src.preprocessing_articles import PreprocessArticles
from src.utils.reader import Reader
from src.utils.writer import Writer

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
