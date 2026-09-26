"""
Candidate generation / blocking for the Amazon ML
Business Entity Resolution Challenge.

This module generates plausible Source-2 / Source-3 candidates
for Source-1 records without performing final matching.

Blocking strategies:
1. Exact normalized name
2. Normalized name core
3. Name token key
4. Country + normalized address number + informative address token
5. Country + informative address token pair

Important:
- Blocking must favor recall.
- Multiple blocking strategies are unioned together.
- Original entity IDs are preserved.
- Country is used as a blocking feature, but never hard-coded.
"""

from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd


REQUIRED_COLUMNS = [
    "entity_id",
    "name_normalized",
    "name_core",
    "address_normalized",
    "country_normalized",
]


# Generic address words are usually poor blocking signals by themselves.
# We keep them in the normalized address, but avoid using them as
# informative address tokens.
GENERIC_ADDRESS_TOKENS = {
    "road",
    "rd",
    "street",
    "st",
    "avenue",
    "ave",
    "drive",
    "dr",
    "lane",
    "ln",
    "boulevard",
    "blvd",
    "highway",
    "hwy",
    "way",
    "unit",
    "apartment",
    "apt",
    "floor",
    "fl",
    "block",
    "plot",
    "po",
    "box",
    "north",
    "south",
    "east",
    "west",
    "northeast",
    "northwest",
    "southeast",
    "southwest",
}


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _tokens(value: str) -> list[str]:
    if not value:
        return []
    return [token for token in value.split() if token]


def _informative_address_tokens(address: str) -> list[str]:
    """
    Return useful address tokens for blocking.

    Generic address words such as 'road', 'street', and 'avenue'
    are excluded because they occur too frequently.

    Tokens are ordered by length so longer/more distinctive tokens
    are preferred.
    """
    tokens = {
        token
        for token in _tokens(address)
        if len(token) >= 4
        and token not in GENERIC_ADDRESS_TOKENS
        and not token.isdigit()
    }

    return sorted(tokens, key=lambda token: (-len(token), token))


def extract_address_numbers(address: str) -> tuple[str, ...]:
    """
    Extract meaningful numeric/alphanumeric address components.

    Leading zeros are removed so that:

        0337 -> 337
        005559 -> 5559
        0017560 -> 17560

    Examples:
        '12 mg road' -> ('12',)
        '024A main street' -> ('24a',)
    """
    if not address:
        return ()

    values: list[str] = []

    for token in address.casefold().split():
        match = re.fullmatch(r"0*(\d+)([a-z])?", token)

        if not match:
            continue

        digits = match.group(1)
        suffix = match.group(2) or ""

        canonical_digits = str(int(digits))

        values.append(f"{canonical_digits}{suffix}")

    return tuple(dict.fromkeys(values))


def _name_token_key(name: str) -> str:
    """
    Create a conservative name-token key.

    Uses up to the first two informative tokens, sorted so that
    simple word-order changes can still share a block.
    """
    tokens = _tokens(name)

    if not tokens:
        return ""

    informative = [
        token
        for token in tokens
        if len(token) >= 3
    ]

    if not informative:
        informative = tokens

    selected = sorted(informative[:3])

    return "|".join(selected[:2])


def _address_pair_key(address: str) -> str:
    """
    Build a country-independent address pair from the two strongest
    informative address tokens.

    Token order is ignored.

    Example:
        '337 Oakland Avenue Michigan City Indiana'
        -> 'michigan|oakland'
    """
    informative = _informative_address_tokens(address)

    if len(informative) < 2:
        return ""

    selected = sorted(informative[:2])

    return "|".join(selected)


def generate_block_keys(row: pd.Series) -> set[str]:
    """
    Generate multiple blocking keys for one normalized record.

    The keys intentionally use several independent signals so that
    one noisy field does not eliminate a true candidate.
    """
    country = str(row.get("country_normalized", "")).strip()
    name = str(row.get("name_normalized", "")).strip()
    name_core = str(row.get("name_core", "")).strip()
    address = str(row.get("address_normalized", "")).strip()

    keys: set[str] = set()

    # ---------------------------------------------------------
    # Name-based blocks
    # ---------------------------------------------------------

    if country and name:
        keys.add(
            f"country_name::{country}::{name}"
        )

    if country and name_core:
        keys.add(
            f"country_core::{country}::{name_core}"
        )

    token_key = _name_token_key(name_core or name)

    if country and token_key:
        keys.add(
            f"country_tokens::{country}::{token_key}"
        )

    # ---------------------------------------------------------
    # Address-number + address-token blocks
    # ---------------------------------------------------------

    address_numbers_value = row.get(
        "address_numbers_normalized",
        "",
    )

    if address_numbers_value:
        address_numbers = tuple(
            token
            for token in str(address_numbers_value).split()
            if token
        )
    else:
        # Backward-compatible fallback for dataframes created
        # before address_numbers_normalized was added.
        address_numbers = extract_address_numbers(address)

    informative_address_tokens = _informative_address_tokens(address)

    if country:
        # Use only the strongest few address tokens to avoid
        # generating excessive candidate blocks.
        for number in address_numbers:
            for token in informative_address_tokens[:3]:
                keys.add(
                    "country_address_number_token::"
                    f"{country}::{number}::{token}"
                )

    # ---------------------------------------------------------
    # Address-token pair block
    # ---------------------------------------------------------

    address_pair = _address_pair_key(address)

    if country and address_pair:
        keys.add(
            f"country_address_pair::{country}::{address_pair}"
        )

    return keys


def build_block_index(
    source_df: pd.DataFrame,
) -> dict[str, list[str]]:
    """
    Build an inverted index:

        blocking_key -> list of entity_ids
    """
    _validate_columns(source_df)

    index: dict[str, list[str]] = defaultdict(list)

    for row in source_df.itertuples(index=False):
        row_dict = row._asdict()

        keys = generate_block_keys(
            pd.Series(row_dict)
        )

        entity_id = str(row_dict["entity_id"])

        for key in keys:
            index[key].append(entity_id)

    return dict(index)


def generate_candidate_pairs(
    source1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    target_source: str,
) -> pd.DataFrame:
    """
    Generate candidate pairs between Source 1 and either
    Source 2 or Source 3.
    """
    _validate_columns(source1_df)
    _validate_columns(target_df)

    if target_source not in {"S2", "S3"}:
        raise ValueError(
            "target_source must be 'S2' or 'S3'"
        )

    index = build_block_index(target_df)

    candidates: set[tuple[str, str, str]] = set()

    for row in source1_df.itertuples(index=False):
        row_dict = row._asdict()

        s1_id = str(row_dict["entity_id"])

        keys = generate_block_keys(
            pd.Series(row_dict)
        )

        for key in keys:
            for candidate_id in index.get(key, []):
                candidates.add(
                    (
                        s1_id,
                        candidate_id,
                        target_source,
                    )
                )

    result = pd.DataFrame(
        list(candidates),
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
            "source",
        ],
    )

    if result.empty:
        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "source",
            ]
        )

    return result.sort_values(
        [
            "source1_entity_id",
            "source",
            "candidate_entity_id",
        ]
    ).reset_index(drop=True)


def generate_candidates_for_both_sources(
    source1_df: pd.DataFrame,
    source2_df: pd.DataFrame,
    source3_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate the union of candidates from Source 2 and Source 3.
    """
    candidates_s2 = generate_candidate_pairs(
        source1_df=source1_df,
        target_df=source2_df,
        target_source="S2",
    )

    candidates_s3 = generate_candidate_pairs(
        source1_df=source1_df,
        target_df=source3_df,
        target_source="S3",
    )

    result = pd.concat(
        [candidates_s2, candidates_s3],
        ignore_index=True,
    )

    return result.drop_duplicates(
        subset=[
            "source1_entity_id",
            "candidate_entity_id",
            "source",
        ]
    ).reset_index(drop=True)