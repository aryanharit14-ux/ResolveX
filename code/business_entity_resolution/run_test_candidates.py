import pandas as pd
import time

from src.config import TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3
from src.data_loader import read_tsv_in_chunks
from src.normalization import normalize_dataframe
from src.candidate_generation import generate_candidate_batch


def load_full(path):
    chunks = list(
        read_tsv_in_chunks(
            path,
            chunk_size=100_000,
        )
    )
    return pd.concat(chunks, ignore_index=True)


start = time.time()

print("Loading test data...")

s1 = load_full(TEST_SOURCE1)
s2 = load_full(TEST_SOURCE2)
s3 = load_full(TEST_SOURCE3)

print(f"S1: {len(s1):,}")
print(f"S2: {len(s2):,}")
print(f"S3: {len(s3):,}")

print()
print("Normalizing test data...")

s1 = normalize_dataframe(s1)
print("S1 normalized")

s2 = normalize_dataframe(s2)
print("S2 normalized")

s3 = normalize_dataframe(s3)
print("S3 normalized")

print(f"Normalization time: {time.time() - start:.2f}s")

print()
print("Generating test candidates...")
print("(This is the expensive stage. Let it run.)")

candidate_batch = generate_candidate_batch(
    source1=s1,
    source2=s2,
    source3=s3,
)

pairs = candidate_batch.pairs

print()
print("TEST CANDIDATES")
print("===============")
print(f"Candidate pairs: {len(pairs):,}")
print(f"S1 entities:     {pairs['source1_entity_id'].nunique():,}")
print(f"S2 candidates:   {(pairs['source'] == 'S2').sum():,}")
print(f"S3 candidates:   {(pairs['source'] == 'S3').sum():,}")
print(f"Total time:      {time.time() - start:.2f}s")

output_path = "../../dataset/test/test_candidates.pkl"

pairs.to_pickle(output_path)

print()
print(f"Saved: {output_path}")