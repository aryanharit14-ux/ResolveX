"""
Training candidate-recall experiment.

Measures how many ground-truth S2/S3 matches are recovered
by the current blocking strategy.

This is an experiment only. It does not create submission files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from business_entity_resolution.src.blocking import (
    generate_candidates_for_both_sources,
)
from business_entity_resolution.src.data_loader import read_tsv_in_chunks
from business_entity_resolution.src.normalization import normalize_dataframe


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CHUNK_SIZE = 50_000

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DATASET_DIR = (
    PROJECT_ROOT
    / "student_resource"
    / "student_resource"
    / "dataset"
    / "train"
)

SOURCE1_PATH = DATASET_DIR / "train_source1.tsv"
SOURCE2_PATH = DATASET_DIR / "train_source2.tsv"
SOURCE3_PATH = DATASET_DIR / "train_source3.tsv"
GROUND_TRUTH_PATH = DATASET_DIR / "train_ground_truth.tsv"


# ---------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------

def load_ground_truth() -> set[tuple[str, str, str]]:
    """
    Convert the ground-truth file into:

        (source1_id, candidate_id, source)

    tuples.
    """

    ground_truth = pd.read_csv(
        GROUND_TRUTH_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    true_pairs: set[tuple[str, str, str]] = set()

    for row in ground_truth.itertuples(index=False):
        source1_id = str(row.source1_entity_id)
        matched_ids = str(row.matched_entity_ids)

        if not matched_ids:
            continue

        for candidate_id in matched_ids.split(","):
            candidate_id = candidate_id.strip()

            if not candidate_id:
                continue

            if candidate_id.startswith("S2-"):
                source = "S2"
            elif candidate_id.startswith("S3-"):
                source = "S3"
            else:
                continue

            true_pairs.add(
                (
                    source1_id,
                    candidate_id,
                    source,
                )
            )

    return true_pairs


# ---------------------------------------------------------------------
# Build normalized target indexes
# ---------------------------------------------------------------------

def load_normalized_source(path: Path) -> pd.DataFrame:
    """
    Load and normalize one complete source.

    Source 2 and Source 3 are large, but this experiment requires
    their blocking indexes to be available simultaneously.
    """

    chunks = []

    for chunk_number, chunk in enumerate(
        read_tsv_in_chunks(
            path,
            chunk_size=CHUNK_SIZE,
        ),
        start=1,
    ):
        print(
            f"  reading {path.name}: chunk {chunk_number}",
            flush=True,
        )

        normalized = normalize_dataframe(chunk)

        chunks.append(normalized)

    return pd.concat(
        chunks,
        ignore_index=True,
    )


# ---------------------------------------------------------------------
# Main recall experiment
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("BLOCKING RECALL EXPERIMENT")
    print("=" * 70)

    print()
    print("Loading ground truth...")

    true_pairs = load_ground_truth()

    print(
        f"True training pairs: {len(true_pairs):,}"
    )

    print()
    print("Loading and normalizing Source 2...")

    source2 = load_normalized_source(
        SOURCE2_PATH
    )

    print(
        f"Source 2 rows: {len(source2):,}"
    )

    print()
    print("Loading and normalizing Source 3...")

    source3 = load_normalized_source(
        SOURCE3_PATH
    )

    print(
        f"Source 3 rows: {len(source3):,}"
    )

    print()
    print("Processing Source 1...")

    recovered_pairs: set[tuple[str, str, str]] = set()

    source1_chunk_number = 0

    for source1_chunk in read_tsv_in_chunks(
        SOURCE1_PATH,
        chunk_size=CHUNK_SIZE,
    ):
        source1_chunk_number += 1

        print(
            f"  Source 1 chunk {source1_chunk_number}",
            flush=True,
        )

        source1_normalized = normalize_dataframe(
            source1_chunk
        )

        candidates = generate_candidates_for_both_sources(
            source1_df=source1_normalized,
            source2_df=source2,
            source3_df=source3,
        )

        for row in candidates.itertuples(index=False):
            recovered_pairs.add(
                (
                    str(row.source1_entity_id),
                    str(row.candidate_entity_id),
                    str(row.source),
                )
            )

    # -------------------------------------------------------------
    # Calculate recall
    # -------------------------------------------------------------

    recovered_true_pairs = true_pairs.intersection(
        recovered_pairs
    )

    missed_pairs = true_pairs - recovered_pairs

    recall = (
        len(recovered_true_pairs)
        / len(true_pairs)
        if true_pairs
        else 0.0
    )

    print()
    print("=" * 70)
    print("BLOCKING RECALL RESULT")
    print("=" * 70)

    print(
        f"True pairs:       {len(true_pairs):,}"
    )

    print(
        f"Recovered pairs:  {len(recovered_true_pairs):,}"
    )

    print(
        f"Missed pairs:     {len(missed_pairs):,}"
    )

    print(
        f"Blocking recall:  {recall * 100:.4f}%"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()