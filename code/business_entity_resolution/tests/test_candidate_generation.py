import pandas as pd

from src.candidate_generation import (
    generate_candidate_batch,
)


def test_candidate_generation_finds_s2_candidate():

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
                "12 mg road delhi"
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
            "entity_id": ["S2-1"],
            "name_normalized": [
                "acme technologies"
            ],
            "name_core": [
                "acme technologies"
            ],
            "address_normalized": [
                "12 mg road delhi"
            ],
            "address_numbers_normalized": [
                "12"
            ],
            "country_normalized": [
                "india"
            ],
        }
    )

    source3 = pd.DataFrame(
        {
            "entity_id": ["S3-1"],
            "name_normalized": [
                "different business"
            ],
            "name_core": [
                "different business"
            ],
            "address_normalized": [
                "99 main street"
            ],
            "address_numbers_normalized": [
                "99"
            ],
            "country_normalized": [
                "india"
            ],
        }
    )

    result = generate_candidate_batch(
        source1,
        source2,
        source3,
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
        &
        (
            result["source"]
            == "S2"
        )
    ).any()


def test_candidate_generation_finds_s3_candidate():

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
                "12 mg road delhi"
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
            "entity_id": ["S2-1"],
            "name_normalized": [
                "different business"
            ],
            "name_core": [
                "different business"
            ],
            "address_normalized": [
                "99 main street"
            ],
            "address_numbers_normalized": [
                "99"
            ],
            "country_normalized": [
                "india"
            ],
        }
    )

    source3 = pd.DataFrame(
        {
            "entity_id": ["S3-1"],
            "name_normalized": [
                "acme technologies"
            ],
            "name_core": [
                "acme technologies"
            ],
            "address_normalized": [
                "12 mg road delhi"
            ],
            "address_numbers_normalized": [
                "12"
            ],
            "country_normalized": [
                "india"
            ],
        }
    )

    result = generate_candidate_batch(
        source1,
        source2,
        source3,
    )

    assert (
        (
            result["source1_entity_id"]
            == "S1-1"
        )
        &
        (
            result["candidate_entity_id"]
            == "S3-1"
        )
        &
        (
            result["source"]
            == "S3"
        )
    ).any()


def test_candidate_generation_contains_both_sources():

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

    source2 = source1.copy()
    source2["entity_id"] = ["S2-1"]

    source3 = source1.copy()
    source3["entity_id"] = ["S3-1"]

    result = generate_candidate_batch(
        source1,
        source2,
        source3,
    )

    assert set(result["source"]) == {
        "S2",
        "S3",
    }


def test_candidate_generation_schema():

    source1 = pd.DataFrame(
        {
            "entity_id": ["S1-1"],
            "name_normalized": ["acme"],
            "name_core": ["acme"],
            "address_normalized": ["12 delhi"],
            "address_numbers_normalized": ["12"],
            "country_normalized": ["india"],
        }
    )

    source2 = source1.copy()
    source2["entity_id"] = ["S2-1"]

    source3 = source1.copy()
    source3["entity_id"] = ["S3-1"]

    result = generate_candidate_batch(
        source1,
        source2,
        source3,
    )

    assert list(result.columns) == [
        "source1_entity_id",
        "candidate_entity_id",
        "source",
    ]

    assert result["source"].isin(
        ["S2", "S3"]
    ).all()