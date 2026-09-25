from pathlib import Path
from typing import Iterator

import pandas as pd


EXPECTED_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


def read_tsv_in_chunks(
    path: Path,
    chunk_size: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """
    Stream a TSV file in chunks.

    Large ResolveX datasets should not be repeatedly loaded
    completely into memory.
    """
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    )


def validate_source_schema(df: pd.DataFrame) -> None:
    """
    Validate the schema of a source dataset.
    """
    missing = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )
