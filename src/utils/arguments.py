from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("-g", "--gui", dest="show_gui", action="store_true", help="show GUI", default=False)
    parser.add_argument(
        "-f", "--force-processing", dest="force_processing", action="store_true", help="force processing", default=False
    )

    parser.add_argument(
        "-n",
        "--number-of-articles",
        dest="number_of_articles",
        help="number of articles to process",
        type=int,
        default=None,
    )

    parser.add_argument("-l", "--lableling", dest="labeling", help="Label preprocessed data", default=None)

    parser.add_argument(
        "-t",
        "--train",
        dest="train",
        action="store_true",
        help="train the threshold for sentiment mapping",
        default=False,
    )

    parser.add_argument(
        "-c", "--compare", dest="compare", action="store_true", help="compare results with labeled data", default=False
    )

    return parser.parse_args()
