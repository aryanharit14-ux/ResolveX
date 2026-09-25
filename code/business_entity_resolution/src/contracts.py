from dataclasses import dataclass

import pandas as pd


CANDIDATE_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "source",
]


@dataclass
class CandidateBatch:
    """
    Candidate pairs produced by the blocking stage.

    source:
        Literal S2 or S3 indicating which source dataset
        produced the candidate entity.
    """

    pairs: pd.DataFrame


@dataclass
class FeatureBatch:
    """
    Numeric features generated for candidate pairs.

    Pair identity and ordering must remain unchanged.
    """

    pairs: pd.DataFrame
    features: pd.DataFrame


@dataclass
class PredictionBatch:
    """
    Model predictions aligned with candidate pairs.
    """

    pairs: pd.DataFrame
    scores: pd.Series
