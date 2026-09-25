from pathlib import Path

from src.output_writer import (
    write_candidate_pairs,
    write_matching_results,
)


def test_write_matching_results(tmp_path: Path):
    output = tmp_path / "matching_results.tsv"

    write_matching_results(
        [
            ("S1-1", ["S2-1", "S3-1", "S2-1"]),
            ("S1-2", []),
        ],
        output,
    )

    lines = output.read_text().splitlines()

    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert lines[1] == "S1-1\tS2-1,S3-1"
    assert lines[2] == "S1-2\t"


def test_write_candidate_pairs(tmp_path: Path):
    output = tmp_path / "candidate_pairs.tsv"

    write_candidate_pairs(
        [
            ("S1-1", ["S2-1", "S3-1", "S2-1"]),
            ("S1-2", []),
        ],
        output,
    )

    lines = output.read_text().splitlines()

    assert lines[0] == "source1_entity_id\tcandidate_entity_ids"
    assert lines[1] == "S1-1\tS2-1,S3-1"
    assert lines[2] == "S1-2\t"
