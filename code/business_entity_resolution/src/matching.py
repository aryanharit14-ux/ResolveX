from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .contracts import FeatureBatch, PredictionBatch


@dataclass
class MatchingModel:
    """
    Pair-level classifier for Source-1 -> Source-2/Source-3 matching.
    """

    model: HistGradientBoostingClassifier

    def predict_scores(self, features: pd.DataFrame) -> pd.Series:
        """
        Return positive-class probabilities aligned with `features`.
        """
        probabilities = self.model.predict_proba(features)[:, 1]
        return pd.Series(probabilities, index=features.index, dtype=float)


def build_matching_model(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> MatchingModel:
    """
    Train the pair-level matching classifier.
    """

    X = X.astype(np.float32)
    y = y.astype(np.int8)

    if len(X) != len(y):
        raise ValueError(
            f"Feature/label length mismatch: {len(X)} != {len(y)}"
        )

    if y.nunique() < 2:
        raise ValueError("Training labels must contain both classes.")

    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=random_state,
    )

    model.fit(X, y)

    return MatchingModel(model=model)


def predict_feature_batch(
    model: MatchingModel,
    feature_batch: FeatureBatch,
) -> PredictionBatch:
    """
    Predict candidate-pair match probabilities while preserving
    candidate ordering.
    """

    scores = model.predict_scores(feature_batch.features)

    if len(scores) != len(feature_batch.pairs):
        raise ValueError(
            "Prediction scores are not aligned with candidate pairs."
        )

    return PredictionBatch(
        pairs=feature_batch.pairs.copy(),
        scores=scores.reset_index(drop=True),
    )


def save_matching_model(
    model: MatchingModel,
    path: str,
) -> None:
    """
    Persist the trained matcher.
    """
    joblib.dump(model, path)


def load_matching_model(
    path: str,
) -> MatchingModel:
    """
    Load a persisted matcher.
    """
    return joblib.load(path)


def build_ground_truth_pairs(
    ground_truth: pd.DataFrame,
) -> set[tuple[str, str]]:
    """
    Convert the ground-truth format

        source1_entity_id
        matched_entity_ids = S2-...,S3-...

    into a set of exact positive pair keys:

        (source1_entity_id, candidate_entity_id)

    Empty matched_entity_ids represent singleton Source-1 entities
    and therefore contribute no positive pairs.
    """

    required = {"source1_entity_id", "matched_entity_ids"}

    missing = required - set(ground_truth.columns)

    if missing:
        raise ValueError(
            f"Ground truth is missing columns: {sorted(missing)}"
        )

    positives: set[tuple[str, str]] = set()

    for row in ground_truth.itertuples(index=False):
        source1_id = str(row.source1_entity_id)
        matched_ids = str(row.matched_entity_ids).strip()

        if not matched_ids:
            continue

        for candidate_id in matched_ids.split(","):
            candidate_id = candidate_id.strip()

            if candidate_id:
                positives.add((source1_id, candidate_id))

    return positives


def add_pair_labels(
    candidate_pairs: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> pd.Series:
    """
    Label candidate pairs using exact ground-truth membership.

    Labels are aligned one-to-one with candidate_pairs.
    """

    required = {
        "source1_entity_id",
        "candidate_entity_id",
        "source",
    }

    missing = required - set(candidate_pairs.columns)

    if missing:
        raise ValueError(
            f"Candidate pairs are missing columns: {sorted(missing)}"
        )

    positives = build_ground_truth_pairs(ground_truth)

    pair_keys = zip(
        candidate_pairs["source1_entity_id"].astype(str),
        candidate_pairs["candidate_entity_id"].astype(str),
    )

    labels = np.fromiter(
        (
            1 if (source1_id, candidate_id) in positives else 0
            for source1_id, candidate_id in pair_keys
        ),
        dtype=np.int8,
        count=len(candidate_pairs),
    )

    return pd.Series(labels, index=candidate_pairs.index, name="label")


def evaluate_candidate_labels(
    candidate_pairs: pd.DataFrame,
    labels: pd.Series,
) -> dict[str, int]:
    """
    Basic sanity statistics for the generated training labels.
    """

    if len(candidate_pairs) != len(labels):
        raise ValueError("Candidate/label length mismatch.")

    positives = int(labels.sum())
    negatives = int(len(labels) - positives)

    return {
        "candidate_pairs": len(candidate_pairs),
        "positive_pairs": positives,
        "negative_pairs": negatives,
    }
