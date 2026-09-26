import pandas as pd

from business_entity_resolution.src.blocking import (
    extract_address_numbers,
    generate_block_keys,
    generate_candidate_pairs,
)


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


def test_candidate_generation_finds_exact_name():
    source1 = pd.DataFrame(
        {
            "entity_id": ["S1-1"],
            "name_normalized": ["acme technologies"],
            "name_core": ["acme technologies"],
            "address_normalized": ["12 mg road"],
            "country_normalized": ["india"],
        }
    )

    source2 = pd.DataFrame(
        {
            "entity_id": ["S2-1", "S2-2"],
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
            result["source1_entity_id"] == "S1-1"
        )
        & (
            result["candidate_entity_id"] == "S2-1"
        )
    ).any()


def test_candidate_generation_does_not_cross_country():
    source1 = pd.DataFrame(
        {
            "entity_id": ["S1-1"],
            "name_normalized": ["acme technologies"],
            "name_core": ["acme technologies"],
            "address_normalized": ["12 mg road"],
            "country_normalized": ["india"],
        }
    )

    source2 = pd.DataFrame(
        {
            "entity_id": ["S2-1"],
            "name_normalized": ["acme technologies"],
            "name_core": ["acme technologies"],
            "address_normalized": ["12 mg road"],
            "country_normalized": ["usa"],
        }
    )

    result = generate_candidate_pairs(
        source1,
        source2,
        "S2",
    )

    assert result.empty