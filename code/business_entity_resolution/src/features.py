from __future__ import annotations

from difflib import SequenceMatcher

import pandas as pd


FEATURE_COLUMNS = [
    "name_exact",
    "name_similarity",
    "address_exact",
    "address_similarity",
    "country_match",
]


def string_similarity(left: str, right: str) -> float:
    """Return a normalized similarity score between two strings."""
    left = "" if left is None else str(left)
    right = "" if right is None else str(right)

    if not left or not right:
        return 0.0

    return SequenceMatcher(None, left, right).ratio()


def build_pair_features(
    source1_row: pd.Series,
    candidate_row: pd.Series,
) -> dict[str, float]:
    """Build similarity features for one S1/candidate pair."""

    name_left = source1_row.get(
        "business_name_normalized",
        source1_row.get("business_name", ""),
    )
    name_right = candidate_row.get(
        "business_name_normalized",
        candidate_row.get("business_name", ""),
    )

    address_left = source1_row.get(
        "business_address_normalized",
        source1_row.get("business_address", ""),
    )
    address_right = candidate_row.get(
        "business_address_normalized",
        candidate_row.get("business_address", ""),
    )

    country_left = source1_row.get(
        "country_normalized",
        source1_row.get("country", ""),
    )
    country_right = candidate_row.get(
        "country_normalized",
        candidate_row.get("country", ""),
    )

    name_left = str(name_left)
    name_right = str(name_right)

    address_left = str(address_left)
    address_right = str(address_right)

    country_left = str(country_left)
    country_right = str(country_right)

    return {
        "name_exact": float(
            bool(name_left) and name_left == name_right
        ),
        "name_similarity": string_similarity(
            name_left,
            name_right,
        ),
        "address_exact": float(
            bool(address_left) and address_left == address_right
        ),
        "address_similarity": string_similarity(
            address_left,
            address_right,
        ),
        "country_match": float(
            bool(country_left)
            and country_left == country_right
        ),
    }
