"""The tracked-rule evaluator (shared/tracked_rules.py).

This drives a UI that a human will use instead of remembering the rules, so the tests are
about the clauses most likely to be forgotten or rationalised away -- the P5-vs-P5
exclusion, the lower edge bound, and an unresolved retractable roof -- plus the constants
still matching the two registrations.
"""

from __future__ import annotations

import pytest

from tracked_rules import (P1_MAX_EDGE, P1_MIN_EDGE, W1_MAX_WIND_MPH, W1_MIN_WIND_MPH,
                           evaluate, evaluate_p1, evaluate_w1)


# --- P1 (NCAA) -------------------------------------------------------------------------

def test_p1_qualifies_and_picks_the_side_from_the_sign():
    over = evaluate_p1(model_total=55.0, market_total=52.0, restricted=True)
    under = evaluate_p1(model_total=49.0, market_total=52.0, restricted=True)
    assert (over.qualifies, over.side) == (True, "OVER")
    assert (under.qualifies, under.side) == (True, "UNDER")


def test_p1_bounds_are_enforced_exactly():
    """0.5 is inclusive, 6.0 is exclusive -- as declared."""
    assert evaluate_p1(model_total=52.5, market_total=52.0, restricted=True).qualifies
    assert not evaluate_p1(model_total=52.4, market_total=52.0, restricted=True).qualifies
    assert not evaluate_p1(model_total=58.0, market_total=52.0, restricted=True).qualifies
    assert evaluate_p1(model_total=57.9, market_total=52.0, restricted=True).qualifies


def test_p1_rejects_p5_vs_p5_even_with_a_perfect_looking_edge():
    """The clause a human is most likely to skip, on exactly the game that tempts them.
    P5-vs-P5 measured 50.1 per 100 -- adding it back turns +39 units into -11."""
    s = evaluate_p1(model_total=55.0, market_total=52.0, restricted=False)
    assert not s.qualifies
    assert "P5" in s.reason, "the reason must name the clause, not just say no"


def test_p1_unknown_universe_is_not_treated_as_eligible():
    s = evaluate_p1(model_total=55.0, market_total=52.0, restricted=None)
    assert not s.qualifies and "unknown" in s.reason.lower()


def test_p1_without_a_market_total_does_not_qualify():
    assert not evaluate_p1(model_total=55.0, market_total=None, restricted=True).qualifies
    assert not evaluate_p1(model_total=float("nan"), market_total=52.0,
                           restricted=True).qualifies


# --- W1 (NFL) --------------------------------------------------------------------------

def test_w1_qualifies_only_at_or_above_the_trigger_and_is_always_under():
    assert evaluate_w1(wind_mph=10.0, roof="outdoors").side == "UNDER"
    assert evaluate_w1(wind_mph=22.0, roof="outdoors").qualifies
    assert not evaluate_w1(wind_mph=9.9, roof="outdoors").qualifies


def test_w1_excludes_indoor_games():
    for roof in ("dome", "closed"):
        assert not evaluate_w1(wind_mph=25.0, roof=roof).qualifies


def test_w1_unresolved_retractable_roof_does_not_qualify():
    """schedules.roof is null for every unplayed game. Unknown is not open -- betting wind
    at a stadium that may be sealed shut is betting on nothing."""
    s = evaluate_w1(wind_mph=25.0, roof=None, venue_roof="retractable")
    assert not s.qualifies and "not open" in s.reason
    # ...but once the state IS published as open, it is an ordinary outdoor game.
    assert evaluate_w1(wind_mph=25.0, roof="open", venue_roof="retractable").qualifies


def test_w1_implausible_wind_is_rejected_not_treated_as_calm():
    s = evaluate_w1(wind_mph=71.0, roof="outdoors")
    assert not s.qualifies and "implausible" in s.reason


def test_w1_missing_forecast_does_not_qualify():
    assert not evaluate_w1(wind_mph=None, roof="outdoors").qualifies


# --- shared behaviour ------------------------------------------------------------------

def test_every_status_carries_a_reason_a_human_can_act_on():
    """A bare 'no' teaches nothing and gets overridden; the reason is the feature."""
    for s in [evaluate_p1(model_total=55.0, market_total=52.0, restricted=False),
              evaluate_p1(model_total=52.1, market_total=52.0, restricted=True),
              evaluate_w1(wind_mph=4.0, roof="outdoors"),
              evaluate_w1(wind_mph=None, roof=None, venue_roof="retractable")]:
        assert s.reason and len(s.reason) > 15
        assert s.label == "—"


def test_label_is_scannable_when_it_qualifies():
    assert evaluate_p1(model_total=55.0, market_total=52.0,
                       restricted=True).label == "P1 · OVER"
    assert evaluate_w1(wind_mph=14.0, roof="outdoors").label == "W1 · UNDER"


def test_dispatch_by_league_and_unknown_leagues_have_no_rule():
    assert evaluate("ncaa", model_total=55.0, market_total=52.0,
                    restricted=True).rule == "P1"
    assert evaluate("nfl", wind_mph=14.0, roof="outdoors").rule == "W1"
    assert not evaluate("xfl").qualifies


def test_constants_match_the_two_registrations():
    """If someone edits these, the UI would start showing a different rule than the one
    being tracked -- silently, and only the log would disagree."""
    assert (P1_MIN_EDGE, P1_MAX_EDGE) == (0.5, 6.0)
    assert (W1_MIN_WIND_MPH, W1_MAX_WIND_MPH) == (10.0, 40.0)
