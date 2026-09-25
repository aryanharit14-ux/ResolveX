import pandas as pd

from src.contracts import (
    CANDIDATE_COLUMNS,
    CandidateBatch,
    FeatureBatch,
    PredictionBatch,
)


def test_candidate_batch_schema():
    pairs = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-2"],
            "candidate_entity_id": ["S2-1", "S3-1"],
            "source": ["S2", "S3"],
        }
    )

    assert list(pairs.columns) == CANDIDATE_COLUMNS
    assert pairs["source"].isin(["S2", "S3"]).all()

    batch = CandidateBatch(pairs=pairs)

    assert len(batch.pairs) == 2


def test_feature_batch_preserves_pair_count():
    pairs = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1"],
            "candidate_entity_id": ["S2-1"],
            "source": ["S2"],
        }
    )

    features = pd.DataFrame(
        {
            "name_similarity": [0.95],
            "address_similarity": [0.80],
        }
    )

    batch = FeatureBatch(
        pairs=pairs,
        features=features,
    )

    assert len(batch.pairs) == len(batch.features)


def test_prediction_batch_preserves_pair_count():
    pairs = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-2"],
            "candidate_entity_id": ["S2-1", "S3-1"],
            "source": ["S2", "S3"],
        }
    )

    scores = pd.Series([0.95, 0.10])

    batch = PredictionBatch(
        pairs=pairs,
        scores=scores,
    )

    assert len(batch.pairs) == len(batch.scores)
