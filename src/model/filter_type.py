from enum import Enum


class FilterType(Enum):
    NONE = 1  # Keep all rows
    PARTIES = 2  # Keep only rows with parties
    SINGLE_PARTY = 3  # Keep only rows with exact one party
