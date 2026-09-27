from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz


FEATURE_COLUMNS = [
    # Name
    "name_exact",
    "name_similarity",
    "name_core_exact",
    "name_core_similarity",
    "name_token_jaccard",
    "name_token_containment",
    "name_length_ratio",

    # Address
    "address_exact",
    "address_similarity",
    "address_token_jaccard",
    "address_token_containment",
    "address_number_similarity",
    "address_length_ratio",

    # Country
    "country_match",

    # Interaction
    "name_address_exact",

    # Missing-data indicators
    "has_name_both",
    "has_address_both",
    "has_country_both",
]


NORMALIZED_COLUMNS = [
    "name_normalized",
    "name_core",
    "address_normalized",
    "address_tokens",
    "address_numbers_normalized",
    "country_normalized",
]


def _as_string_series(series: pd.Series) -> pd.Series:
    """Convert a pandas Series to safe strings."""
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )


def _length_ratio(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    """Vectorized shorter/longer string length ratio."""

    left_len = left.str.len()
    right_len = right.str.len()

    longer = np.maximum(left_len, right_len)
    shorter = np.minimum(left_len, right_len)

    return pd.Series(
        np.divide(
            shorter,
            longer,
            out=np.zeros(len(left), dtype=float),
            where=longer.to_numpy() != 0,
        ),
        index=left.index,
    )


def _token_jaccard(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    """
    Token Jaccard similarity.

    J(A,B) = |A intersection B| / |A union B|
    """

    left_sets = left.str.split().map(set)
    right_sets = right.str.split().map(set)

    values = []

    for a, b in zip(left_sets, right_sets):
        if not a or not b:
            values.append(0.0)
            continue

        union = a | b

        if not union:
            values.append(0.0)
        else:
            values.append(len(a & b) / len(union))

    return pd.Series(values, index=left.index, dtype=float)


def _token_containment(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    """
    Token containment.

    Measures how much of the smaller token set occurs
    in the larger token set.
    """

    left_sets = left.str.split().map(set)
    right_sets = right.str.split().map(set)

    values = []

    for a, b in zip(left_sets, right_sets):
        if not a or not b:
            values.append(0.0)
            continue

        smaller = a if len(a) <= len(b) else b
        larger = b if len(a) <= len(b) else a

        values.append(len(smaller & larger) / len(smaller))

    return pd.Series(values, index=left.index, dtype=float)


def _number_similarity(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    """
    Compare normalized address-number tokens as sets.

    Example:

        "12 15" vs "15 12" -> 1.0

    Unlike raw string equality, ordering does not matter.
    """

    left_sets = left.str.split().map(set)
    right_sets = right.str.split().map(set)

    values = []

    for a, b in zip(left_sets, right_sets):
        if not a or not b:
            values.append(0.0)
            continue

        union = a | b

        if not union:
            values.append(0.0)
        else:
            values.append(len(a & b) / len(union))

    return pd.Series(values, index=left.index, dtype=float)


def _rapidfuzz_ratio(
    left: pd.Series,
    right: pd.Series,
) -> np.ndarray:
    """
    Calculate pairwise RapidFuzz similarity.

    RapidFuzz's fuzz.ratio returns 0-100.
    Convert to 0-1.
    """

    return np.fromiter(
        (
            fuzz.ratio(a, b) / 100.0
            if a and b
            else 0.0
            for a, b in zip(left, right)
        ),
        dtype=np.float64,
        count=len(left),
    )


def _validate_normalized_columns(
    dataframe: pd.DataFrame,
    dataframe_name: str,
) -> None:
    missing = [
        column
        for column in NORMALIZED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{dataframe_name} is missing normalized columns: {missing}"
        )


def build_pair_features(
    source1_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build pairwise features for two already-aligned DataFrames.

    Row i of source1_df is compared with row i of candidate_df.

    This function does NOT perform entity-ID lookups.
    """

    if len(source1_df) != len(candidate_df):
        raise ValueError(
            "source1_df and candidate_df must have the same number of rows."
        )

    _validate_normalized_columns(source1_df, "source1_df")
    _validate_normalized_columns(candidate_df, "candidate_df")

    s1_name = _as_string_series(
        source1_df["name_normalized"]
    )

    candidate_name = _as_string_series(
        candidate_df["name_normalized"]
    )

    s1_name_core = _as_string_series(
        source1_df["name_core"]
    )

    candidate_name_core = _as_string_series(
        candidate_df["name_core"]
    )

    s1_address = _as_string_series(
        source1_df["address_normalized"]
    )

    candidate_address = _as_string_series(
        candidate_df["address_normalized"]
    )

    s1_address_tokens = _as_string_series(
        source1_df["address_tokens"]
    )

    candidate_address_tokens = _as_string_series(
        candidate_df["address_tokens"]
    )

    s1_address_numbers = _as_string_series(
        source1_df["address_numbers_normalized"]
    )

    candidate_address_numbers = _as_string_series(
        candidate_df["address_numbers_normalized"]
    )

    s1_country = _as_string_series(
        source1_df["country_normalized"]
    )

    candidate_country = _as_string_series(
        candidate_df["country_normalized"]
    )

    # ---------------------------------------------------------
    # Exact matches
    # ---------------------------------------------------------

    name_exact = (
        (s1_name != "")
        & (candidate_name != "")
        & (s1_name == candidate_name)
    ).astype(float)

    name_core_exact = (
        (s1_name_core != "")
        & (candidate_name_core != "")
        & (s1_name_core == candidate_name_core)
    ).astype(float)

    address_exact = (
        (s1_address != "")
        & (candidate_address != "")
        & (s1_address == candidate_address)
    ).astype(float)

    country_match = (
        (s1_country != "")
        & (candidate_country != "")
        & (s1_country == candidate_country)
    ).astype(float)

    # ---------------------------------------------------------
    # Fuzzy name features
    # ---------------------------------------------------------

    name_similarity = _rapidfuzz_ratio(
        s1_name,
        candidate_name,
    )

    name_core_similarity = _rapidfuzz_ratio(
        s1_name_core,
        candidate_name_core,
    )

    # ---------------------------------------------------------
    # Name token features
    # ---------------------------------------------------------

    name_token_jaccard = _token_jaccard(
        s1_name,
        candidate_name,
    )

    name_token_containment = _token_containment(
        s1_name,
        candidate_name,
    )

    name_length_ratio = _length_ratio(
        s1_name,
        candidate_name,
    )

    # ---------------------------------------------------------
    # Address fuzzy features
    # ---------------------------------------------------------

    address_similarity = _rapidfuzz_ratio(
        s1_address,
        candidate_address,
    )

    address_token_jaccard = _token_jaccard(
        s1_address_tokens,
        candidate_address_tokens,
    )

    address_token_containment = _token_containment(
        s1_address_tokens,
        candidate_address_tokens,
    )

    address_number_similarity = _number_similarity(
        s1_address_numbers,
        candidate_address_numbers,
    )

    address_length_ratio = _length_ratio(
        s1_address,
        candidate_address,
    )

    # ---------------------------------------------------------
    # Interaction
    # ---------------------------------------------------------

    name_address_exact = (
        (name_exact == 1.0)
        & (address_exact == 1.0)
    ).astype(float)

    # ---------------------------------------------------------
    # Missing-data indicators
    # ---------------------------------------------------------

    has_name_both = (
        (s1_name != "")
        & (candidate_name != "")
    ).astype(float)

    has_address_both = (
        (s1_address != "")
        & (candidate_address != "")
    ).astype(float)

    has_country_both = (
        (s1_country != "")
        & (candidate_country != "")
    ).astype(float)

    return pd.DataFrame(
        {
            "name_exact": name_exact,
            "name_similarity": name_similarity,
            "name_core_exact": name_core_exact,
            "name_core_similarity": name_core_similarity,
            "name_token_jaccard": name_token_jaccard,
            "name_token_containment": name_token_containment,
            "name_length_ratio": name_length_ratio,

            "address_exact": address_exact,
            "address_similarity": address_similarity,
            "address_token_jaccard": address_token_jaccard,
            "address_token_containment": address_token_containment,
            "address_number_similarity": address_number_similarity,
            "address_length_ratio": address_length_ratio,

            "country_match": country_match,

            "name_address_exact": name_address_exact,

            # Missing-data indicators
            "has_name_both": has_name_both,
            "has_address_both": has_address_both,
            "has_country_both": has_country_both,
        },
        index=source1_df.index,
    )


def build_feature_dataframe(
    candidate_pairs: pd.DataFrame,
    source1_df: pd.DataFrame,
    source2_df: pd.DataFrame,
    source3_df: pd.DataFrame,
    batch_size: int = 100_000,
) -> pd.DataFrame:
    """
    Build features for the complete candidate set.

    Candidate ordering is preserved exactly.

    Processing is performed in batches to avoid creating
    extremely large intermediate DataFrames.
    """

    required = {
        "source1_entity_id",
        "candidate_entity_id",
        "source",
    }

    missing = required - set(candidate_pairs.columns)

    if missing:
        raise ValueError(
            f"candidate_pairs missing columns: {sorted(missing)}"
        )

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    _validate_normalized_columns(source1_df, "source1_df")
    _validate_normalized_columns(source2_df, "source2_df")
    _validate_normalized_columns(source3_df, "source3_df")

    s1 = source1_df.set_index("entity_id", drop=False)
    s2 = source2_df.set_index("entity_id", drop=False)
    s3 = source3_df.set_index("entity_id", drop=False)

    results: list[pd.DataFrame] = []

    for start in range(0, len(candidate_pairs), batch_size):

        end = min(
            start + batch_size,
            len(candidate_pairs),
        )

        batch = candidate_pairs.iloc[start:end].copy()

        # -----------------------------------------------------
        # Source-1 lookup
        # -----------------------------------------------------

        source1_rows = s1.loc[
            batch["source1_entity_id"].to_numpy()
        ].reset_index(drop=True)

        # -----------------------------------------------------
        # Split S2/S3 candidates
        # -----------------------------------------------------

        source2_mask = batch["source"].eq("S2")
        source3_mask = batch["source"].eq("S3")

        if (~(source2_mask | source3_mask)).any():
            invalid_sources = sorted(
                batch.loc[
                    ~(source2_mask | source3_mask),
                    "source",
                ].unique()
            )

            raise ValueError(
                f"Unknown candidate source values: {invalid_sources}"
            )

        candidate_rows = pd.DataFrame(
            index=range(len(batch)),
            columns=source2_df.columns,
        )

        # -----------------------------------------------------
        # S2 lookup
        # -----------------------------------------------------

        if source2_mask.any():
            ids = batch.loc[
                source2_mask,
                "candidate_entity_id",
            ].to_numpy()

            rows = s2.loc[ids].reset_index(drop=True)

            candidate_rows.loc[
                source2_mask.to_numpy(),
                rows.columns,
            ] = rows.to_numpy()

        # -----------------------------------------------------
        # S3 lookup
        # -----------------------------------------------------

        if source3_mask.any():
            ids = batch.loc[
                source3_mask,
                "candidate_entity_id",
            ].to_numpy()

            rows = s3.loc[ids].reset_index(drop=True)

            candidate_rows.loc[
                source3_mask.to_numpy(),
                rows.columns,
            ] = rows.to_numpy()

        candidate_rows = candidate_rows.reset_index(drop=True)

        # Convert object columns safely back to strings.
        for column in NORMALIZED_COLUMNS:
            candidate_rows[column] = _as_string_series(
                candidate_rows[column]
            )

        features = build_pair_features(
            source1_rows,
            candidate_rows,
        )

        results.append(features)

    if not results:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    final_features = pd.concat(
        results,
        ignore_index=True,
    )

    if len(final_features) != len(candidate_pairs):
        raise RuntimeError(
            "Feature row count does not match candidate row count: "
            f"{len(final_features)} != {len(candidate_pairs)}"
        )

    return final_features[FEATURE_COLUMNS]