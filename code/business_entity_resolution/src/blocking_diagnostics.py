from collections import Counter, defaultdict
import sys

import pandas as pd

from src.normalization import normalize_chunks
from src.blocking import (
    _build_address_token_frequency,
    _build_address_number_frequency,
    _build_name_token_frequency,
    _build_name_ngram_frequency,
    _build_address_pair_frequency,
    _build_address_number_location_frequency,
    _build_leetspeak_name_frequency,
    generate_block_keys,
    build_block_index,
)


def load_sample(path, n):
    df = pd.read_csv(
        path,
        sep="\t",
        nrows=n,
        dtype=str,
        keep_default_na=False,
    )

    normalized = normalize_chunks(iter([df]))
    return pd.concat(normalized, ignore_index=True)


def analyze(source1_df, target_df, target_name):
    print("\n" + "=" * 90)
    print(f"TARGET: {target_name}")
    print(f"S1 rows:     {len(source1_df):,}")
    print(f"Target rows: {len(target_df):,}")
    print("=" * 90)

    # Same frequency statistics used by the real candidate generator.
    address_token_frequency = _build_address_token_frequency(target_df)
    address_number_frequency = _build_address_number_frequency(target_df)
    name_token_frequency = _build_name_token_frequency(target_df)
    name_ngram_frequency = _build_name_ngram_frequency(target_df)
    address_pair_frequency = _build_address_pair_frequency(target_df)
    address_number_location_frequency = (
        _build_address_number_location_frequency(target_df)
    )
    leetspeak_name_frequency = _build_leetspeak_name_frequency(target_df)

    # Build target block index using exactly the same configuration.
    index = build_block_index(
        target_df,
        address_token_frequency=address_token_frequency,
        address_number_frequency=address_number_frequency,
        name_token_frequency=name_token_frequency,
        name_ngram_frequency=name_ngram_frequency,
        address_pair_frequency=address_pair_frequency,
        address_number_location_frequency=address_number_location_frequency,
        leetspeak_name_frequency=leetspeak_name_frequency,
    )

    # ------------------------------------------------------------
    # Target-side block sizes
    # ------------------------------------------------------------

    block_sizes = {
        key: len(values)
        for key, values in index.items()
    }

    print("\nTOP 25 LARGEST BLOCKS")
    print("-" * 90)

    for key, size in sorted(
        block_sizes.items(),
        key=lambda x: -x[1],
    )[:25]:
        print(f"{size:>8,}  {key}")

    # ------------------------------------------------------------
    # Strategy-level statistics
    # ------------------------------------------------------------

    stats = defaultdict(
        lambda: {
            "blocks": 0,
            "target_rows": 0,
            "max_block": 0,
            "s1_hits": 0,
            "candidate_pairs": 0,
        }
    )

    for key, size in block_sizes.items():
        strategy = key.split("::", 1)[0]

        stats[strategy]["blocks"] += 1
        stats[strategy]["target_rows"] += size
        stats[strategy]["max_block"] = max(
            stats[strategy]["max_block"],
            size,
        )

    # ------------------------------------------------------------
    # Measure actual S1 -> target candidate contribution
    # ------------------------------------------------------------

    for row in source1_df.itertuples(index=False):
        row_series = pd.Series(row._asdict())

        keys = generate_block_keys(
            row_series,
            address_token_frequency=address_token_frequency,
            address_number_frequency=address_number_frequency,
            name_token_frequency=name_token_frequency,
            name_ngram_frequency=name_ngram_frequency,
            address_pair_frequency=address_pair_frequency,
            address_number_location_frequency=(
                address_number_location_frequency
            ),
            leetspeak_name_frequency=leetspeak_name_frequency,
        )

        for key in keys:
            target_ids = index.get(key, [])

            if not target_ids:
                continue

            strategy = key.split("::", 1)[0]

            stats[strategy]["s1_hits"] += 1
            stats[strategy]["candidate_pairs"] += len(target_ids)

    print("\nSTRATEGY CONTRIBUTION")
    print("-" * 90)
    print(
        f"{'Strategy':40s}"
        f"{'Blocks':>10s}"
        f"{'MaxBlock':>10s}"
        f"{'S1Hits':>12s}"
        f"{'Candidates':>18s}"
    )

    for strategy, data in sorted(
        stats.items(),
        key=lambda x: -x[1]["candidate_pairs"],
    ):
        print(
            f"{strategy:40s}"
            f"{data['blocks']:>10,}"
            f"{data['max_block']:>10,}"
            f"{data['s1_hits']:>12,}"
            f"{data['candidate_pairs']:>18,}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python -m src.blocking_diagnostics <sample_size>"
        )
        sys.exit(1)

    sample_size = int(sys.argv[1])

    print(f"Loading first {sample_size:,} rows from each source...")

    source1 = load_sample(
        "../../dataset/train/train_source1.tsv",
        sample_size,
    )

    source2 = load_sample(
        "../../dataset/train/train_source2.tsv",
        sample_size,
    )

    source3 = load_sample(
        "../../dataset/train/train_source3.tsv",
        sample_size,
    )

    analyze(source1, source2, "S2")
    analyze(source1, source3, "S3")
