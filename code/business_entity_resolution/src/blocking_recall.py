from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.blocking import generate_candidates_for_both_sources
from src.normalization import normalize_dataframe


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    """
    Load the ground-truth matching pairs.

    Ground-truth format:

        source1_entity_id    matched_entity_ids

    The comma-separated matched_entity_ids are expanded into
    one source1/candidate pair per match.
    """

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "source1_entity_id",
        "matched_entity_ids",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Ground-truth file is missing columns: {sorted(missing)}"
        )

    rows = []

    for row in df.itertuples(index=False):
        source1_id = str(row.source1_entity_id)
        matched_ids = str(row.matched_entity_ids)

        if not matched_ids:
            continue

        for candidate_id in matched_ids.split(","):
            candidate_id = candidate_id.strip()

            if candidate_id:
                rows.append(
                    (
                        source1_id,
                        candidate_id,
                    )
                )

    return pd.DataFrame(
        rows,
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
        ],
    ).drop_duplicates()


def calculate_blocking_recall(
    candidate_pairs: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict:
    """
    Calculate blocking recall.

    Recall =
        recovered true pairs
        -------------------
        total true pairs
    """

    candidate_set = set(
        zip(
            candidate_pairs["source1_entity_id"].astype(str),
            candidate_pairs["candidate_entity_id"].astype(str),
        )
    )

    truth_set = set(
        zip(
            ground_truth["source1_entity_id"].astype(str),
            ground_truth["candidate_entity_id"].astype(str),
        )
    )

    recovered = candidate_set.intersection(truth_set)
    missed = truth_set - candidate_set

    total_true = len(truth_set)
    recovered_count = len(recovered)

    recall = (
        recovered_count / total_true
        if total_true > 0
        else 0.0
    )

    return {
        "total_true_pairs": total_true,
        "recovered_pairs": recovered_count,
        "missed_pairs": len(missed),
        "blocking_recall": recall,
        "candidate_pairs": len(candidate_set),

        # Kept internally for diagnostic output.
        "_missed_pair_ids": missed,
    }


def print_missed_pairs(
    missed_pairs,
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    limit: int = 20,
) -> None:
    """
    Print details of missed true pairs so we can understand
    why the blocking rules failed to generate them.
    """

    source1_lookup = (
        source1
        .drop_duplicates("entity_id")
        .set_index("entity_id")
    )

    target = pd.concat(
        [source2, source3],
        ignore_index=True,
    )

    target_lookup = (
        target
        .drop_duplicates("entity_id")
        .set_index("entity_id")
    )

    print("\n" + "=" * 70)
    print(
        f"FIRST {min(limit, len(missed_pairs))} "
        "MISSED TRUE PAIRS"
    )
    print("=" * 70)

    for i, (source1_id, candidate_id) in enumerate(
        sorted(missed_pairs)[:limit],
        start=1,
    ):
        print(f"\n--- Missed Pair {i} ---")

        print(f"Source 1 ID    : {source1_id}")
        print(f"True Match ID  : {candidate_id}")

        if source1_id in source1_lookup.index:
            s1 = source1_lookup.loc[source1_id]

            print(
                f"S1 Name        : "
                f"{s1.get('business_name', '')}"
            )

            print(
                f"S1 Name Norm   : "
                f"{s1.get('name_normalized', '')}"
            )

            print(
                f"S1 Name Core   : "
                f"{s1.get('name_core', '')}"
            )

            print(
                f"S1 Address     : "
                f"{s1.get('business_address', '')}"
            )

            print(
                f"S1 Address Norm: "
                f"{s1.get('address_normalized', '')}"
            )

            print(
                f"S1 Country     : "
                f"{s1.get('country', '')}"
            )

        if candidate_id in target_lookup.index:
            target_row = target_lookup.loc[candidate_id]

            print(
                f"Target Name    : "
                f"{target_row.get('business_name', '')}"
            )

            print(
                f"Target Name Norm: "
                f"{target_row.get('name_normalized', '')}"
            )

            print(
                f"Target Name Core: "
                f"{target_row.get('name_core', '')}"
            )

            print(
                f"Target Address : "
                f"{target_row.get('business_address', '')}"
            )

            print(
                f"Target Address Norm: "
                f"{target_row.get('address_normalized', '')}"
            )

            print(
                f"Target Country : "
                f"{target_row.get('country', '')}"
            )

    print("\n" + "=" * 70)


def filter_ground_truth_for_sample(
    ground_truth: pd.DataFrame,
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
) -> pd.DataFrame:
    """
    Restrict ground truth to entities that are actually present
    in the development sample.
    """

    source1_ids = set(
        source1["entity_id"].astype(str)
    )

    target_ids = set(
        pd.concat(
            [
                source2["entity_id"],
                source3["entity_id"],
            ],
            ignore_index=True,
        ).astype(str)
    )

    filtered = ground_truth[
        ground_truth["source1_entity_id"].astype(str).isin(source1_ids)
        &
        ground_truth["candidate_entity_id"].astype(str).isin(target_ids)
    ].copy()

    return filtered.drop_duplicates()


def evaluate_blocking(
    source1_path: str | Path,
    source2_path: str | Path,
    source3_path: str | Path,
    ground_truth_path: str | Path,
    sample_size: int | None = None,
    show_misses: int = 0,
) -> dict:

    print("Loading source files...")

    read_kwargs = {
        "sep": "\t",
        "dtype": str,
        "keep_default_na": False,
    }

    if sample_size is not None:
        read_kwargs["nrows"] = sample_size

        print(
            f"Development mode: loading first "
            f"{sample_size:,} rows from each source."
        )

    source1 = pd.read_csv(
        source1_path,
        **read_kwargs,
    )

    source2 = pd.read_csv(
        source2_path,
        **read_kwargs,
    )

    source3 = pd.read_csv(
        source3_path,
        **read_kwargs,
    )

    print("Normalizing source files...")

    source1 = normalize_dataframe(source1)
    source2 = normalize_dataframe(source2)
    source3 = normalize_dataframe(source3)

    print(f"Source 1 rows: {len(source1):,}")
    print(f"Source 2 rows: {len(source2):,}")
    print(f"Source 3 rows: {len(source3):,}")

    print("\nGenerating candidates...")

    candidates = generate_candidates_for_both_sources(
        source1_df=source1,
        source2_df=source2,
        source3_df=source3,
    )

    print(
        f"Candidate pairs generated: "
        f"{len(candidates):,}"
    )

    print("\nLoading ground truth...")

    ground_truth = load_ground_truth(
        ground_truth_path
    )

    if sample_size is not None:
        ground_truth = filter_ground_truth_for_sample(
            ground_truth=ground_truth,
            source1=source1,
            source2=source2,
            source3=source3,
        )

        print(
            "Ground truth restricted to "
            "development sample."
        )

    print(
        f"True pairs: "
        f"{len(ground_truth):,}"
    )

    results = calculate_blocking_recall(
        candidate_pairs=candidates,
        ground_truth=ground_truth,
    )

    # Diagnostic output for missed pairs.
    if show_misses > 0:
        print_missed_pairs(
            missed_pairs=results["_missed_pair_ids"],
            source1=source1,
            source2=source2,
            source3=source3,
            limit=show_misses,
        )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate blocking recall."
    )

    parser.add_argument(
        "--source1",
        required=True,
        help="Path to source1 TSV",
    )

    parser.add_argument(
        "--source2",
        required=True,
        help="Path to source2 TSV",
    )

    parser.add_argument(
        "--source3",
        required=True,
        help="Path to source3 TSV",
    )

    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Path to ground truth TSV",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "Load only the first N rows from each source "
            "for development evaluation."
        ),
    )

    parser.add_argument(
        "--show-misses",
        type=int,
        default=0,
        help="Print details of the first N missed true pairs.",
    )

    args = parser.parse_args()

    results = evaluate_blocking(
        source1_path=args.source1,
        source2_path=args.source2,
        source3_path=args.source3,
        ground_truth_path=args.ground_truth,
        sample_size=args.sample_size,
        show_misses=args.show_misses,
    )

    print("\n" + "=" * 50)
    print("BLOCKING EVALUATION")
    print("=" * 50)

    print(
        f"True pairs       : "
        f"{results['total_true_pairs']:,}"
    )

    print(
        f"Recovered pairs  : "
        f"{results['recovered_pairs']:,}"
    )

    print(
        f"Missed pairs     : "
        f"{results['missed_pairs']:,}"
    )

    print(
        f"Candidate pairs  : "
        f"{results['candidate_pairs']:,}"
    )

    print(
        f"Blocking Recall  : "
        f"{results['blocking_recall']:.2%}"
    )

    print("=" * 50)