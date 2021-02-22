from pandas import DataFrame


class Writer:
    """
    Class that writes a dataframe to a json file.
    """

    @staticmethod
    def write_dataframe(dataframe: DataFrame, filename: str) -> None:
        """
        Helper function to store Pandas dataframe into json file

        :param dataframe: the Pandas dataframe which should be stored in json
        :param filename: the path where the dataframe should be stored
        """
        path = "src/output/" + filename + ".json"
        with open(path, "w", encoding="utf-8") as file:
            dataframe.to_json(file, force_ascii=False, orient="records", default_handler=str, index=True)
