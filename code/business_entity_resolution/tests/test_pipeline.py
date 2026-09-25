import pandas as pd

from src.contracts import (
    CandidateBatch,
    FeatureBatch,
    PredictionBatch,
)
from src.pipeline import build_outputs, run_pipeline
from src.output_writer import (
    write_candidate_pairs as real_write_candidate_pairs,
    write_matching_results as real_write_matching_results,
)


def test_pipeline_preserves_pair_alignment(tmp_path, monkeypatch):
    candidate_pairs = pd.DataFrame(
        {
            "source1_entity_id": [
                "S1-1",
                "S1-1",
                "S1-2",
            ],
            "candidate_entity_id": [
                "S2-1",
                "S3-1",
                "S2-2",
            ],
            "source": [
                "S2",
                "S3",
                "S2",
            ],
        }
    )

    candidate_batch = CandidateBatch(pairs=candidate_pairs)

    def feature_fn(batch):
        features = pd.DataFrame(
            {
                "name_similarity": [0.95, 0.20, 0.90],
                "address_similarity": [0.90, 0.10, 0.80],
            }
        )

        return FeatureBatch(
            pairs=batch.pairs.copy(),
            features=features,
        )

    def predict_fn(batch):
        scores = pd.Series(
            [0.97, 0.15, 0.91],
            dtype=float,
        )

        return PredictionBatch(
            pairs=batch.pairs.copy(),
            scores=scores,
        )

    result = run_pipeline(
        candidate_batch=candidate_batch,
        feature_fn=feature_fn,
        predict_fn=predict_fn,
        threshold=0.80,
    )

    assert len(result.candidate_batch.pairs) == 3
    assert len(result.feature_batch.features) == 3
    assert len(result.prediction_batch.scores) == 3

    assert (
        result.prediction_batch.pairs["candidate_entity_id"].tolist()
        == ["S2-1", "S3-1", "S2-2"]
    )

    import src.pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module,
        "write_candidate_pairs",
        lambda rows: real_write_candidate_pairs(
            rows,
            tmp_path / "candidate_pairs.tsv",
        ),
    )

    monkeypatch.setattr(
        pipeline_module,
        "write_matching_results",
        lambda rows: real_write_matching_results(
            rows,
            tmp_path / "matching_results.tsv",
        ),
    )

    build_outputs(result, threshold=0.80)

    matching = (
        tmp_path / "matching_results.tsv"
    ).read_text().splitlines()

    candidates = (
        tmp_path / "candidate_pairs.tsv"
    ).read_text().splitlines()

    assert matching[0] == (
        "source1_entity_id\tmatched_entity_ids"
    )
    assert matching[1] == "S1-1\tS2-1"
    assert matching[2] == "S1-2\tS2-2"

    assert candidates[0] == (
        "source1_entity_id\tcandidate_entity_ids"
    )
    assert candidates[1] == "S1-1\tS2-1,S3-1"
    assert candidates[2] == "S1-2\tS2-2"
