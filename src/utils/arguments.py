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
        action="store_true",
        help="number of articles to process",
        default=None,
    )

    parser.add_argument(
        "-w",
        "--write",
        dest="write",
        action="store_true",
        help="write processed dataframe to json (overwrite existing dataframe)",
        default=True,
    )

    parser.add_argument(
        "-l", "--lableling", dest="labeling", action="store_true", help="Label preprocessed data", default=False
    )

    parser.add_argument(
        "-c", "--compare", dest="compare", action="store_true", help="compare results with labeled data", default=False
    )

    return parser.parse_args()
