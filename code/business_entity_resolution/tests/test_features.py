import pandas as pd

from src.features import (
    FEATURE_COLUMNS,
    build_pair_features,
    string_similarity,
)


def test_string_similarity_identical():
    assert string_similarity("abc", "abc") == 1.0


def test_string_similarity_empty():
    assert string_similarity("", "abc") == 0.0


def test_build_pair_features():
    source1 = pd.Series(
        {
            "business_name_normalized": "abc corp",
            "business_address_normalized": "123 main st",
            "country_normalized": "india",
        }
    )

    candidate = pd.Series(
        {
            "business_name_normalized": "abc corp",
            "business_address_normalized": "123 main st",
            "country_normalized": "india",
        }
    )

    features = build_pair_features(source1, candidate)

    assert features["name_exact"] == 1.0
    assert features["name_similarity"] == 1.0
    assert features["address_exact"] == 1.0
    assert features["address_similarity"] == 1.0
    assert features["country_match"] == 1.0


def test_feature_columns_exist():
    source1 = pd.Series(
        {
            "business_name": "ABC Corp",
            "business_address": "123 Main St",
            "country": "India",
        }
    )

    candidate = pd.Series(
        {
            "business_name": "ABC Corporation",
            "business_address": "123 Main Street",
            "country": "India",
        }
    )

    features = build_pair_features(source1, candidate)

    assert set(features.keys()) == set(FEATURE_COLUMNS)
