import pandas as pd

from src.blocking import (
    extract_address_numbers,
    generate_block_keys,
    generate_candidate_pairs,
)


# ============================================================
# Address number tests
# ============================================================

def test_extract_address_numbers():
    result = extract_address_numbers(
        "12 MG Road Bengaluru"
    )

    assert result == ("12",)


def test_extract_alphanumeric_address_number():
    result = extract_address_numbers(
        "24a Main Street"
    )

    assert result == ("24a",)


# ============================================================
# Blocking key tests
# ============================================================

def test_block_keys_include_country_name():
    row = pd.Series(
        {
            "entity_id": "S1-1",
            "name_normalized": "acme technologies pvt ltd",
            "name_core": "acme technologies",
            "address_normalized": "12 mg road",
            "country_normalized": "india",
        }
    )

    keys = generate_block_keys(row)

    assert (
        "country_name::india::acme technologies pvt ltd"
        in keys
    )

    assert (
        "country_core::india::acme technologies"
        in keys
    )


def test_exact_name_without_country_is_recall_fallback():
    """
    Exact name should still produce a blocking key when
    country information is missing.
    """

    row = pd.Series(
        {
            "entity_id": "S1-1",
            "name_normalized": "acme technologies",
            "name_core": "acme technologies",
            "address_normalized": "",
            "country_normalized": "",
        }
    )

    keys = generate_block_keys(row)

    assert (
        "name::acme technologies"
        in keys
    )


def test_address_number_block():
    """
    Address number + informative address token should
    produce a country-aware blocking key.
    """

    row = pd.Series(
        {
            "entity_id": "S1-1",
            "name_normalized": "different business",
            "name_core": "different business",
            "address_normalized": "337 oakland avenue",
            "address_numbers_normalized": "337",
            "country_normalized": "usa",
        }
    )

    keys = generate_block_keys(row)

    assert (
        "country_address_number_token::"
        "usa::337::oakland"
        in keys
    )


def test_name_core_block():
    """
    Name core should create a country-aware blocking key.
    """

    row = pd.Series(
        {
            "entity_id": "S1-1",
            "name_normalized": "acme technologies pvt ltd",
            "name_core": "acme technologies",
            "address_normalized": "",
            "country_normalized": "india",
        }
    )

    keys = generate_block_keys(row)

    assert (
        "country_core::india::acme technologies"
        in keys
    )


# ============================================================
# Candidate generation tests
# ============================================================

def test_candidate_generation_finds_exact_name():
    source1 = pd.DataFrame(
        {
            "entity_id": ["S1-1"],
            "name_normalized": [
                "acme technologies"
            ],
            "name_core": [
                "acme technologies"
            ],
            "address_normalized": [
                "12 mg road"
            ],
            "address_numbers_normalized": [
                "12"
            ],
            "country_normalized": [
                "india"
            ],
        }
    )

    source2 = pd.DataFrame(
        {
            "entity_id": [
                "S2-1",
                "S2-2",
            ],
            "name_normalized": [
                "acme technologies",
                "different business",
            ],
            "name_core": [
                "acme technologies",
                "different business",
            ],
            "address_normalized": [
                "12 mg road",
                "99 main street",
            ],
            "address_numbers_normalized": [
                "12",
                "99",
            ],
            "country_normalized": [
                "india",
                "india",
            ],
        }
    )

    result = generate_candidate_pairs(
        source1,
        source2,
        "S2",
    )

    assert (
        (
            result["source1_entity_id"]
            == "S1-1"
        )
        &
        (
            result["candidate_entity_id"]
            == "S2-1"
        )
    ).any()


def test_country_aware_block_does_not_cross_country():
    """
    Country-aware blocking keys must not match records
    from different countries.

    The implementation also contains country-independent
    fallback keys for recall, so the complete candidate
    generator may still produce a candidate.
    """

    source1 = pd.Series(
        {
            "entity_id": "S1-1",
            "name_normalized": "acme technologies",
            "name_core": "acme technologies",
            "address_normalized": "12 mg road",
            "address_numbers_normalized": "12",
            "country_normalized": "india",
        }
    )

    source2 = pd.Series(
        {
            "entity_id": "S2-1",
            "name_normalized": "acme technologies",
            "name_core": "acme technologies",
            "address_normalized": "12 mg road",
            "address_numbers_normalized": "12",
            "country_normalized": "usa",
        }
    )

    source1_keys = generate_block_keys(source1)
    source2_keys = generate_block_keys(source2)

    source1_country_keys = {
        key
        for key in source1_keys
        if key.startswith("country_")
    }

    source2_country_keys = {
        key
        for key in source2_keys
        if key.startswith("country_")
    }

    assert source1_country_keys.isdisjoint(
        source2_country_keys
    )