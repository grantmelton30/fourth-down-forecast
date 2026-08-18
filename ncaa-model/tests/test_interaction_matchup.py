"""A genuine offense x defense interaction term, not just split ratings (GATES.md,
follow-up to the pass/rush EPA test).

The pass/rush test asked whether rush-only ratings deserve their own *linear* term --
they don't, pooled beat them standalone. This is a different claim: whether the *effect*
of facing a worse defense scales with how good the offense already is, a multiplicative
relationship the model's existing additive formula cannot represent regardless of how the
underlying ratings are computed. `interaction_matchup` is the one new piece; whether either
version (pooled or rush-only) actually promotes is a question for a real rebuild, not a
unit test -- this file only pins the arithmetic.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import load_config
from src.ratings import interaction_matchup


def _games():
    return pd.DataFrame([{
        "game_id": 1, "season": 2025, "week": 1, "homeTeam": "A", "awayTeam": "B",
    }])


def _walkforward(team_off_def: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {"season": 2025, "week": 1, "team": team, "off_rating": o, "def_rating": d}
        for team, (o, d) in team_off_def.items()
    ])


def test_interaction_matchup_matches_hand_computed_product_diff_and_sum():
    cfg = load_config()  # unused by interaction_matchup, kept for parity with sibling tests
    games = _games()  # A home vs B away
    wf = _walkforward({"A": (0.30, -0.10), "B": (0.10, 0.05)})

    out = interaction_matchup(wf, games, prefix="pooled_")
    row = out.iloc[0]

    home_interaction = 0.30 * 0.05  # A's offense * B's defense
    away_interaction = 0.10 * -0.10  # B's offense * A's defense
    assert row.pooled_interaction_diff == pytest.approx(home_interaction - away_interaction)
    assert row.pooled_interaction_sum == pytest.approx(home_interaction + away_interaction)


def test_interaction_matchup_is_a_product_not_the_linear_net_epa_combination():
    """The whole point of this feature: it must NOT reduce to net_epa_vec's weighted sum."""
    games = _games()
    wf = _walkforward({"A": (1.0, 1.0), "B": (1.0, 1.0)})
    out = interaction_matchup(wf, games, prefix="pooled_")
    row = out.iloc[0]
    # home=away=1*1=1 each way; diff must be exactly 0 (a product, not a weighted sum that
    # would generally be nonzero here), sum must be exactly 2.
    assert row.pooled_interaction_diff == pytest.approx(0.0)
    assert row.pooled_interaction_sum == pytest.approx(2.0)


def test_interaction_matchup_respects_the_prefix_and_is_challenger_ready():
    games = _games()
    wf = _walkforward({"A": (0.0, 0.0), "B": (0.0, 0.0)})
    out = interaction_matchup(wf, games, prefix="rush_")
    assert list(out.columns) == ["game_id", "rush_interaction_diff", "rush_interaction_sum"]
