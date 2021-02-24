from enum import Enum


class PlotType(Enum):
    STATISTICS = 0
    SENTIMENT_PARTY = 1
    SENTIMENT_OUTLET = 2
    TOPICS_PARTY = 3
    TOPICS_MEDIA = 4
    TOPICS_PARTY_MEDIA = 5
    TIME_COURSE = 6
