from __future__ import annotations

import pandas as pd
import pytest

from feature_snapshot import FeatureSnapshot, merge_asof_features


def test_snapshot_rejects_information_available_after_kickoff():
    with pytest.raises(ValueError, match="available_at"):
        FeatureSnapshot(
            league="nfl", entity="CHI", feature="qb_continuity", value=1.0,
            available_at="2026-09-13T18:00:00Z", source="nflverse-depth-chart",
            game_id="2026_01_GB_CHI", kickoff="2026-09-13T17:00:00Z",
        )


def test_asof_merge_uses_latest_pre_cutoff_value_only():
    games = pd.DataFrame(
        {"game_id": ["g"], "team": ["A"],
         "data_cutoff": ["2026-09-01T12:00:00Z"]}
    )
    features = pd.DataFrame(
        {"team": ["A", "A"], "feature": ["talent", "talent"],
         "value": [0.4, 0.9],
         "available_at": ["2026-08-01T12:00:00Z", "2026-09-02T12:00:00Z"],
         "source": ["cfbd-talent", "cfbd-talent"]}
    )

    out = merge_asof_features(games, features, entity_column="team")
    assert out.loc[0, "talent"] == pytest.approx(0.4)
    assert out.loc[0, "talent__source"] == "cfbd-talent"


def test_missing_features_remain_missing_and_are_reported():
    games = pd.DataFrame(
        {"game_id": ["g"], "team": ["A"],
         "data_cutoff": ["2026-09-01T12:00:00Z"]}
    )
    features = pd.DataFrame(
        columns=["team", "feature", "value", "available_at", "source"]
    )
    out = merge_asof_features(
        games, features, entity_column="team", required_features=["injury_burden"]
    )
    assert pd.isna(out.loc[0, "injury_burden"])
    assert out.loc[0, "unavailable_features"] == ("injury_burden",)
