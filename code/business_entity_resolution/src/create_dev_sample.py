from pathlib import Path
import random

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
DEV_DIR = PROJECT_ROOT / "dataset" / "dev"

GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"
SOURCE1 = TRAIN_DIR / "train_source1.tsv"
SOURCE2 = TRAIN_DIR / "train_source2.tsv"
SOURCE3 = TRAIN_DIR / "train_source3.tsv"

SEED = 42
S1_SAMPLE_SIZE = 5_000
DISTRACTORS_PER_SOURCE = 20_000
CHUNK_SIZE = 100_000


def read_sample_ground_truth():
    """
    Deterministically sample Source-1 entities from the training ground truth.
    """
    rng = random.Random(SEED)

    reservoir = []

    with GROUND_TRUTH.open("r", encoding="utf-8") as f:
        header = next(f)

        for index, line in enumerate(f):
            line = line.rstrip("\n")

            if not line:
                continue

            parts = line.split("\t", 1)

            if len(parts) != 2:
                continue

            s1_id, matched_ids = parts

            if len(reservoir) < S1_SAMPLE_SIZE:
                reservoir.append((s1_id, matched_ids))
            else:
                j = rng.randint(0, index)

                if j < S1_SAMPLE_SIZE:
                    reservoir[j] = (s1_id, matched_ids)

    reservoir.sort(key=lambda x: x[0])

    return reservoir


def collect_target_ids(sample_ground_truth):
    """
    Collect all S2/S3 IDs that are true matches for the sampled S1 entities.
    """
    source2_ids = set()
    source3_ids = set()

    for _, matched in sample_ground_truth:
        if not matched:
            continue

        for entity_id in matched.split(","):
            entity_id = entity_id.strip()

            if entity_id.startswith("S2-"):
                source2_ids.add(entity_id)

            elif entity_id.startswith("S3-"):
                source3_ids.add(entity_id)

    return source2_ids, source3_ids


def write_filtered_source(
    source_path,
    output_path,
    required_ids,
    distractor_count,
):
    """
    Keep every required true-match record plus a deterministic set of
    additional records as distractors.
    """
    required_ids = set(required_ids)

    kept = set()
    rows = []

    with source_path.open("r", encoding="utf-8") as f:
        header = next(f)
        rows.append(header)

        for line in f:
            if len(kept) >= distractor_count and required_ids.issubset(kept):
                break

            parts = line.rstrip("\n").split("\t", 1)

            if not parts:
                continue

            entity_id = parts[0]

            # Always retain true matches.
            if entity_id in required_ids:
                rows.append(line)
                kept.add(entity_id)
                continue

            # Deterministic distractors: retain the first records until
            # the requested number has been reached.
            if len(kept) < distractor_count:
                rows.append(line)
                kept.add(entity_id)

    # The logic above can stop before discovering required IDs that occur
    # later. Scan again if necessary.
    missing = required_ids - kept

    if missing:
        with source_path.open("r", encoding="utf-8") as f:
            next(f)

            for line in f:
                entity_id = line.split("\t", 1)[0]

                if entity_id in missing:
                    rows.append(line)
                    kept.add(entity_id)
                    missing.remove(entity_id)

                    if not missing:
                        break

    output_path.write_text(
        "".join(rows),
        encoding="utf-8",
    )

    return len(rows) - 1


def write_source1(sample_ids, output_path):
    sample_ids = set(sample_ids)

    chunks = []

    for chunk in pd.read_csv(
        SOURCE1,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):
        selected = chunk[chunk["entity_id"].isin(sample_ids)]

        if not selected.empty:
            chunks.append(selected)

    result = pd.concat(chunks, ignore_index=True)

    # Preserve deterministic S1 ordering.
    result["_order"] = result["entity_id"].map(
        {entity_id: i for i, entity_id in enumerate(sorted(sample_ids))}
    )

    result = (
        result
        .sort_values("_order")
        .drop(columns="_order")
    )

    result.to_csv(
        output_path,
        sep="\t",
        index=False,
        encoding="utf-8",
    )

    return len(result)


def main():
    DEV_DIR.mkdir(parents=True, exist_ok=True)

    print("Creating deterministic ResolveX development sample...")
    print(f"Seed: {SEED}")
    print(f"S1 sample size: {S1_SAMPLE_SIZE}")

    sample_ground_truth = read_sample_ground_truth()

    sample_s1_ids = [
        s1_id for s1_id, _ in sample_ground_truth
    ]

    source2_ids, source3_ids = collect_target_ids(
        sample_ground_truth
    )

    print(f"Selected S1 entities: {len(sample_s1_ids)}")
    print(f"Required S2 matches: {len(source2_ids)}")
    print(f"Required S3 matches: {len(source3_ids)}")

    s1_count = write_source1(
        sample_s1_ids,
        DEV_DIR / "dev_source1.tsv",
    )

    print(f"Wrote S1 records: {s1_count}")

    s2_count = write_filtered_source(
        SOURCE2,
        DEV_DIR / "dev_source2.tsv",
        source2_ids,
        DISTRACTORS_PER_SOURCE,
    )

    print(f"Wrote S2 records: {s2_count}")

    s3_count = write_filtered_source(
        SOURCE3,
        DEV_DIR / "dev_source3.tsv",
        source3_ids,
        DISTRACTORS_PER_SOURCE,
    )

    print(f"Wrote S3 records: {s3_count}")

    ground_truth_df = pd.DataFrame(
        sample_ground_truth,
        columns=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    )

    ground_truth_df.to_csv(
        DEV_DIR / "dev_ground_truth.tsv",
        sep="\t",
        index=False,
        encoding="utf-8",
    )

    print(
        f"Wrote ground truth: {len(ground_truth_df)} rows"
    )

    print("Development sample created successfully.")


if __name__ == "__main__":
    main()
