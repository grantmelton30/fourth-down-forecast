import pandas as pd
import pytest

from src.evaluation import eligible_rows, chronological_split


def test_future_opportunities_cannot_change_earlier_eligibility():
    f = pd.DataFrame({"player_id": ["p"]*4, "kickoff": pd.date_range("2024-01-01", periods=4),
                      "targets": [1, 2, 5, 100], "season": [2023, 2024, 2025, 2026]})
    first = eligible_rows(f, "targets", 2).index.tolist()
    f.loc[3, "targets"] = 0
    assert eligible_rows(f, "targets", 2).index.tolist() == first
    train, test = chronological_split(f, 2025)
    assert train.season.tolist() == [2023, 2024]
    assert test.season.tolist() == [2025]


def test_empty_training_is_rejected():
    with pytest.raises(ValueError):
        chronological_split(pd.DataFrame({"season": [2025]}), 2025)
