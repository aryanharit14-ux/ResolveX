from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .contracts import CandidateBatch, FeatureBatch, PredictionBatch
from .output_writer import (
    write_candidate_pairs,
    write_matching_results,
)


@dataclass
class PipelineResult:
    candidate_batch: CandidateBatch
    feature_batch: FeatureBatch
    prediction_batch: PredictionBatch


def run_pipeline(
    candidate_batch: CandidateBatch,
    feature_fn: Callable[[CandidateBatch], FeatureBatch],
    predict_fn: Callable[[FeatureBatch], PredictionBatch],
    threshold: float,
) -> PipelineResult:
    """
    Run the complete candidate -> features -> prediction pipeline.

    The candidate pair ordering must remain unchanged through feature
    generation and prediction.
    """

    feature_batch = feature_fn(candidate_batch)

    if len(feature_batch.pairs) != len(candidate_batch.pairs):
        raise ValueError(
            "Feature stage changed the number of candidate pairs."
        )

    prediction_batch = predict_fn(feature_batch)

    if len(prediction_batch.pairs) != len(feature_batch.pairs):
        raise ValueError(
            "Prediction stage changed the number of candidate pairs."
        )

    if len(prediction_batch.scores) != len(prediction_batch.pairs):
        raise ValueError(
            "Prediction scores are not aligned with candidate pairs."
        )

    return PipelineResult(
        candidate_batch=candidate_batch,
        feature_batch=feature_batch,
        prediction_batch=prediction_batch,
    )


def build_outputs(
    result: PipelineResult,
    threshold: float,
) -> None:
    """
    Convert pair-level candidates and predictions into the two required
    submission files.
    """

    pairs = result.candidate_batch.pairs.copy()
    scores = result.prediction_batch.scores.reset_index(drop=True)

    if len(pairs) != len(scores):
        raise ValueError(
            "Cannot build outputs: pair and score counts differ."
        )

    pairs = pairs.reset_index(drop=True)
    pairs["score"] = scores

    # Every candidate passed to the matching model is written here.
    candidate_rows = []

    for source1_id, group in pairs.groupby(
        "source1_entity_id",
        sort=False,
    ):
        candidate_rows.append(
            (
                source1_id,
                group["candidate_entity_id"].tolist(),
            )
        )

    write_candidate_pairs(candidate_rows)

    # Only predictions at or above the threshold become final matches.
    matched_rows = []

    for source1_id, group in pairs.groupby(
        "source1_entity_id",
        sort=False,
    ):
        matches = group.loc[
            group["score"] >= threshold,
            "candidate_entity_id",
        ].tolist()

        matched_rows.append(
            (
                source1_id,
                matches,
            )
        )

    write_matching_results(matched_rows)
