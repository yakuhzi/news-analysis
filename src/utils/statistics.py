from typing import Dict, List, Tuple

from pandas import DataFrame


class Statistics:
    @staticmethod
    def get_basic_statistics(dataframe: DataFrame, media: List[str], parties: List[str]) -> Tuple[List[str], List[int]]:
        """
        Get the basic statistic (document distribution).

        :param dataframe: Dataframe to extract the statistics from.
        :param parties: Parties to consider for the statistics.
        :param media: Media to consider for the statistics.

        :return: Tuple containing the document distributions.
        """
        x = ["Total"]
        y = [len(dataframe)]

        for media in media:
            media_documents = dataframe[dataframe["media"] == media]
            y.append(len(media_documents))
            x.append(media)

        for party in parties:
            party_documents = dataframe[dataframe.apply(lambda row: party in row["parties"], axis=1)]
            y.append(len(party_documents))
            x.append(party)

        return x, y

    @staticmethod
    def get_media_statistics(dataframe: DataFrame, media: str, parties: List[str]) -> Tuple[List[str], List[int]]:
        """
        Get the basic statistic (document distribution) of each media.

        :param dataframe: Dataframe to extract the statistics from.
        :param parties: Parties to consider for the statistics.
        :param media: Media to consider for the statistics.

        :return: Tuple containing the document distributions.
        """
        media_documents = dataframe[dataframe["media"] == media]

        x = ["Total"]
        y = [len(media_documents)]

        for party in parties:
            party_documents = media_documents[media_documents.apply(lambda row: party in row["parties"], axis=1)]
            y.append(len(party_documents))
            x.append(party)

        return x, y

    @staticmethod
    def get_party_statistics(dataframe: DataFrame, party: str, media: List[str]) -> Tuple[List[str], List[int]]:
        """
        Get the basic statistic (document distribution) of each party.

        :param dataframe: Dataframe to extract the statistics from.
        :param parties: Parties to consider for the statistics.
        :param media: Media to consider for the statistics.

        :return: Tuple containing the document distributions.
        """
        party_documents = dataframe[dataframe.apply(lambda row: party in row["parties"], axis=1)]

        x = ["Total"]
        y = [len(party_documents)]

        for media in media:
            media_documents = party_documents[party_documents["media"] == media]
            y.append(len(media_documents))
            x.append(media)

        return x, y

    @staticmethod
    def get_sentiment_statistics(
        df_paragraphs: DataFrame, by_party: bool, parties: List[str], media: List[str]
    ) -> Dict[str, Dict[str, Tuple[int, int, int]]]:
        """
        Get statistics for the sentiment either grouped by party or by media outlet.

        :param df_paragraphs: the dataframe of the paragraphs
        :param by_party: If True, group data by party, otherwise group by media
        :param parties: List of parties to consider. Defaults to all parties.
        :param media: List of media outlets to consider. Defaults to all media outlets.

        :return: Dictionary containing the statistics
        """
        sentiment_statistics: Dict[str, Dict[str, Tuple[int, int, int]]] = {}

        # Define parties for grouping
        if parties is None:
            parties = ["CDU", "CSU", "SPD", "AfD", "Grüne", "Linke"]

        # Define media for grouping
        if media is None:
            media = ["Tagesschau", "TAZ", "Bild"]

        # Iterate over parties or media
        for item_1 in parties if by_party else media:
            party_statistics: Dict[str, Tuple[int, int, int]] = {}

            # Iterate over media or parties
            for item_2 in media if by_party else parties:
                # Get paragraphs filtered by party and media
                df_paragraphs_filtered = df_paragraphs[
                    (df_paragraphs["media"] == (item_2 if by_party else item_1))
                    & (df_paragraphs["parties"].apply(lambda row: (item_1 if by_party else item_2) in row))
                ]

                # Get number of paragraphs filtered by party and media
                total_number = len(df_paragraphs_filtered)

                if total_number == 0:
                    party_statistics[item_2] = (0, 0, 0)
                    continue

                # Get number of positive, negative and neutral sentences
                positive = len(df_paragraphs_filtered.loc[(df_paragraphs_filtered["sentiment"] == "Positive")])
                negative = len(df_paragraphs_filtered.loc[(df_paragraphs_filtered["sentiment"] == "Negative")])
                neutral = len(df_paragraphs_filtered.loc[(df_paragraphs_filtered["sentiment"] == "Neutral")])

                party_statistics[item_2] = (positive, negative, neutral)

            sentiment_statistics[item_1] = party_statistics

        print(sentiment_statistics)
        return sentiment_statistics
