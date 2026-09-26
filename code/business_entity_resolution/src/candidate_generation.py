from __future__ import annotations

import pandas as pd

from src.blocking import generate_candidates_for_both_sources


def generate_candidate_batch(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate candidate pairs between Source 1 and
    Sources 2 and 3.

    Returns a DataFrame with:

        source1_entity_id
        candidate_entity_id
        source
    """

    return generate_candidates_for_both_sources(
        source1_df=source1,
        source2_df=source2,
        source3_df=source3,
    )