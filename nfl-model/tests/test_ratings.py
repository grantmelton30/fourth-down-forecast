"""Ratings correctness (§5), above all the sign convention.

§5a calls the sign flip "the single most common bug in this kind of build", so it gets an
explicit test against a season whose best and worst defenses are not in dispute.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import load_config
from src.ingest import CANONICAL_TEAMS, load_pbp, load_schedules, normalize_team
from src.ratings import (
    build_game_offense_table,
    fit_pace,
    fit_ratings,
    net_epa,
    season_priors,
)


@pytest.fixture(scope="module")
def fitted():
    cfg = load_config()
    schedules = load_schedules(cfg.train_seasons)
    pbp = load_pbp(cfg.train_seasons)
    game_off = build_game_offense_table(pbp, schedules, cache_key="main")
    # End of the 2024 regular season and playoffs.
    fit = fit_ratings(game_off, pd.Timestamp("2025-02-15"), cfg)
    return cfg, schedules, game_off, fit


def test_all_teams_present_and_centered(fitted):
    _, _, _, fit = fitted
    assert list(fit.table.index) == CANONICAL_TEAMS
    assert fit.table["off_rating"].mean() == pytest.approx(0.0, abs=1e-9)
    assert fit.table["def_rating"].mean() == pytest.approx(0.0, abs=1e-9)


def test_def_rating_sign_convention(fitted):
    """Higher def_rating = WORSE defense.

    2024's best EPA defenses were Philadelphia, Minnesota, Baltimore, Houston and Kansas
    City; its worst were Carolina and Jacksonville. The best of those must sit below the
    worst.
    """
    _, _, _, fit = fitted
    best = fit.table.nsmallest(8, "def_rating").index
    worst = fit.table.nlargest(8, "def_rating").index
    assert "PHI" in best, f"PHI should be a top defense, got {list(best)}"
    assert "CAR" in worst, f"CAR should be a bottom defense, got {list(worst)}"
    assert fit.table.loc["PHI", "def_rating"] < fit.table.loc["CAR", "def_rating"]


def test_off_rating_sign_convention(fitted):
    """Higher off_rating = BETTER offense. Baltimore, Buffalo and Detroit led 2024."""
    _, _, _, fit = fitted
    top = fit.table.nlargest(8, "off_rating").index
    assert {"BAL", "BUF", "DET"} & set(top)
    assert fit.table.loc["BAL", "off_rating"] > fit.table.loc["CAR", "off_rating"]


def test_net_epa_rewards_facing_a_worse_defense(fitted):
    """The load-bearing consequence of the convention.

    An offense facing a *worse* defense (higher def_rating) must project higher, not
    lower. The playbook's literal `off - def` formula inverts this; see the note in
    ratings.net_epa and DECISIONS.md.
    """
    cfg, _, _, fit = fitted
    off = fit.table.loc["BAL", "off_rating"]
    good_defense = fit.table["def_rating"].min()
    bad_defense = fit.table["def_rating"].max()
    assert net_epa(off, bad_defense, cfg) > net_epa(off, good_defense, cfg)


def test_ratings_are_plausible_magnitude(fitted):
    """EPA per play: best-to-worst spread is a few tenths, not whole units."""
    _, _, _, fit = fitted
    assert 0.05 < fit.table["off_rating"].max() < 0.5
    assert 0.05 < fit.table["def_rating"].max() < 0.5


def test_pace_ratings_centered_and_small(fitted):
    cfg, _, game_off, _ = fitted
    pace = fit_pace(game_off, pd.Timestamp("2025-02-15"), cfg)
    assert pace.mean() == pytest.approx(0.0, abs=1e-9)
    # Pace deviations are fractions of a drive per game, not whole drives.
    assert pace.abs().max() < 2.0


def test_season_priors_regress_toward_zero(fitted):
    cfg, _, _, fit = fitted
    priors = season_priors(fit, cfg)
    keep = 1.0 - cfg.ratings.offseason_regression
    assert priors["prior_off"].loc["BAL"] == pytest.approx(
        fit.table["off_rating"].loc["BAL"] * keep
    )
    assert priors["prior_off"].abs().max() < fit.table["off_rating"].abs().max()


def test_venue_hfa_deviations_are_centered(fitted):
    """PATCH 01 §0 Bug A: the league baseline belongs to the drive multinomial, so what
    context.py contributes must be a pure deviation averaging zero across all 32 venues."""
    from src.context import estimate_venue_hfa
    from src.ratings import build_walkforward_ratings

    cfg, schedules, game_off, _ = fitted
    wf = build_walkforward_ratings(game_off, schedules, cfg)
    venue = estimate_venue_hfa(schedules, wf, cfg)
    assert abs(venue.mean_deviation) < 0.1, (
        f"per-venue HFA deviations average {venue.mean_deviation:+.3f}, not ~0 -- the "
        "league baseline has leaked back into context.py"
    )
    # A deviation is a modest correction, not a second home-field advantage.
    assert venue.by_team.abs().max() < 2.0


def test_drive_start_is_near_own_27():
    """PATCH 01 §0 Bug C: drive start must come from the first *scrimmage* play.

    Reading it off the kickoff row puts the mean near midfield; the true average drive
    starts around a team's own 27, i.e. yardline_100 ~= 73.
    """
    from src.drives import build_drive_table

    cfg = load_config()
    pbp = load_pbp(cfg.train_seasons)
    drives = build_drive_table(pbp, cache_key="main")
    mean_start = float(drives["start_yardline_100"].mean())
    assert 68.0 < mean_start < 78.0, (
        f"mean drive start is yardline_100={mean_start:.1f}; near 50 means the kickoff "
        "row is being read as the drive's first play"
    )


def test_normalize_team_maps_relocations():
    assert normalize_team("OAK") == "LV"
    assert normalize_team("SD") == "LAC"
    assert normalize_team("STL") == "LA"
    assert normalize_team("LAR") == "LA"
    assert normalize_team("JAC") == "JAX"
    assert normalize_team("KC") == "KC"


def test_spread_line_sign_convention():
    """nflverse: positive spread_line = home favored, aligned with result.

    DATA_SOURCES.md §9 lists a flipped spread sign as the single most common silent bug,
    so it is asserted against a known blowout rather than assumed.
    """
    cfg = load_config()
    schedules = load_schedules(cfg.train_seasons)
    played = schedules[schedules["result"].notna() & schedules["spread_line"].notna()]
    # Detroit were double-digit home favorites over Jacksonville in 2024 and won by 46.
    game = played[played["game_id"] == "2024_11_JAX_DET"]
    if len(game):
        assert game["spread_line"].iloc[0] > 0
        assert game["result"].iloc[0] > 0
    assert played["result"].corr(played["spread_line"]) > 0.3


def test_home_offense_is_half_at_neutral_sites():
    cfg = load_config()
    schedules = load_schedules(cfg.train_seasons)
    pbp = load_pbp(cfg.train_seasons)
    game_off = build_game_offense_table(pbp, schedules, cache_key="main")
    neutral_ids = schedules.loc[schedules["location"] == "Neutral", "game_id"]
    neutral = game_off[game_off["game_id"].isin(neutral_ids)]
    if len(neutral):
        assert (neutral["home_offense"] == 0.5).all()
