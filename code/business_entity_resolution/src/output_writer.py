from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

from .config import MATCHING_RESULTS, CANDIDATE_PAIRS


def write_matching_results(
    rows: Iterable[Tuple[str, Iterable[str]]],
    output_path: Path = MATCHING_RESULTS,
) -> None:
    """
    Write final Source-1 -> Source-2/Source-3 matches.

    Each Source-1 entity gets exactly one row.
    matched_entity_ids are comma-separated.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []

    for source1_id, matched_ids in rows:
        ids = list(dict.fromkeys(str(x) for x in matched_ids))
        records.append(
            {
                "source1_entity_id": str(source1_id),
                "matched_entity_ids": ",".join(ids),
            }
        )

    df = pd.DataFrame(
        records,
        columns=["source1_entity_id", "matched_entity_ids"],
    )

    df.to_csv(
        output_path,
        sep="\t",
        index=False,
        encoding="utf-8",
    )


def write_candidate_pairs(
    rows: Iterable[Tuple[str, Iterable[str]]],
    output_path: Path = CANDIDATE_PAIRS,
) -> None:
    """
    Write the final candidate set passed to the matching model.

    Each Source-1 entity gets exactly one row.
    candidate_entity_ids are comma-separated.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []

    for source1_id, candidate_ids in rows:
        ids = list(dict.fromkeys(str(x) for x in candidate_ids))
        records.append(
            {
                "source1_entity_id": str(source1_id),
                "candidate_entity_ids": ",".join(ids),
            }
        )

    df = pd.DataFrame(
        records,
        columns=["source1_entity_id", "candidate_entity_ids"],
    )

    df.to_csv(
        output_path,
        sep="\t",
        index=False,
        encoding="utf-8",
    )
