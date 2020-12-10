from utils.reader import Reader

if __name__ == "__main__":
    reader = Reader()
    reader.read_articles()

    print("Number of Bild articles: {}".format(len(reader.bild_articles)))
    print("Number of Tagesschau articles: {}".format(len(reader.tagesschau_articles)))
    print("Number of TAZ articles: {}".format(len(reader.taz_articles)))
