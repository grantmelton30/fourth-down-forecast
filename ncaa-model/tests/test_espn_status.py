"""ESPN quarterback availability (DECISIONS.md D17).

This feed can move a real line by 1.75 points, in a season that has not started, off an
undocumented API. So the tests here are mostly about what it must REFUSE to do: it must not
invent an absence from an ambiguous status, must not touch a game whose box score already
settles the question, and must say so out loud when it has nothing.
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

from src import espn
from src.config import load_config
from src.qb import format_status_report, qb_ratings_table, resolve_status


def test_inactive_is_not_an_absence():
    """REGRESSION. Every graduated passer reads `Inactive` -- it means "not on an active
    roster", which for a former player is permanent and says nothing about this week.
    Scoring it as unavailable produced 66 spurious absences on the first live test."""
    assert espn._status_is_available("Inactive") is True
    assert espn._status_is_available("Active") is True


def test_unambiguous_absences_are_absences():
    for status in ("Out", "Suspended", "Injured", "Ineligible"):
        assert espn._status_is_available(status) is False, status


def test_questionable_is_treated_as_playing():
    """Mirrors nfl-model/src/injuries.py: most questionable players play, and a false
    absence moves the line the full 1.75 in the wrong direction."""
    assert espn._status_is_available("Questionable") is True
    assert espn._status_is_available("Day-To-Day") is True


def test_an_unseen_status_is_None_and_gets_reported():
    """Nothing but Active/Inactive has been observed -- the season has not started. The
    first real injury week will produce codes this module has never seen, and they must
    surface rather than being silently treated as either thing."""
    assert espn._status_is_available("Limited Participation") is None
    frame = pd.DataFrame([{"athlete_id": "1", "status": "Limited Participation",
                           "available": None, "fetched_at": time.time()}])
    assert espn.unrecognised_statuses(frame) == ["Limited Participation"]
    # ...and an unknown status must NOT produce an override.
    table = pd.DataFrame([{"game_id": 1, "season": 2026, "week": 1, "team": "A",
                           "incumbent_id": "1"}])
    assert espn.qb_status_overrides(table, frame).empty


def test_overrides_are_built_only_from_confirmed_absences():
    status = pd.DataFrame([
        {"athlete_id": "out1", "status": "Out", "available": False,
         "fetched_at": time.time()},
        {"athlete_id": "ok1", "status": "Active", "available": True,
         "fetched_at": time.time()},
    ])
    table = pd.DataFrame([
        {"game_id": 1, "season": 2026, "week": 1, "team": "A", "incumbent_id": "out1"},
        {"game_id": 1, "season": 2026, "week": 1, "team": "B", "incumbent_id": "ok1"},
    ])
    out = espn.qb_status_overrides(table, status)
    assert list(out["team"]) == ["A"]
    assert bool(out["starter_out"].iloc[0]) is True
    assert out["source"].iloc[0].startswith("espn:")


def test_offline_fetch_makes_no_network_call(tmp_path, monkeypatch):
    monkeypatch.setattr(espn, "CACHE_DIR", tmp_path)

    def _explode(*a, **k):  # noqa: ANN002, ANN003
        raise AssertionError("network used with allow_network=False")

    monkeypatch.setattr(espn.requests, "get", _explode)
    got = espn.fetch_athlete_status(["123"], allow_network=False)
    assert got.empty


def test_a_played_game_is_never_overridden_by_the_feed():
    """History is settled by its box score. Letting a live feed rewrite a completed game is
    how 66 graduated quarterbacks became retroactive absences."""
    cfg = load_config()
    passers = pd.DataFrame([
        {"game_id": 1, "season": 2025, "week": 1, "team": "A", "qb_id": "starter",
         "qb_name": "S", "attempts": 30, "ppa": 0.3},
        {"game_id": 2, "season": 2025, "week": 2, "team": "A", "qb_id": "starter",
         "qb_name": "S", "attempts": 30, "ppa": 0.3},
    ])
    schedule = pd.DataFrame([
        {"game_id": 1, "season": 2025, "week": 1, "homeTeam": "A", "awayTeam": "Z"},
        {"game_id": 2, "season": 2025, "week": 2, "homeTeam": "A", "awayTeam": "Z"},
        {"game_id": 3, "season": 2025, "week": 3, "homeTeam": "A", "awayTeam": "Z"},
    ])
    table = qb_ratings_table(passers, cfg, games=schedule)
    assert bool(table["played"].any()) and not bool(table["played"].all())

    played_before = table[table["played"]]["incumbent_absent"].sum()
    # manual_dir=None must be tolerated: a caller with no manual directory is a
    # normal state, not a crash.
    out, report = resolve_status(table, None, allow_network=False)
    assert out[out["played"]]["incumbent_absent"].sum() == played_before
    # Only the unplayed week 3 game is eligible to be asked about.
    assert report["upcoming_games"] >= 1


def test_status_report_says_plainly_when_it_has_nothing():
    """"No absences" and "the feed is broken" look identical in a projection unless the
    run says which one it is."""
    line = format_status_report(
        {"espn_checked": 0, "espn_absences": 0, "manual_absences": 0, "unrecognised": []})
    assert "no availability data" in line
    loud = format_status_report(
        {"espn_checked": 5, "espn_absences": 0, "manual_absences": 0,
         "unrecognised": ["Weird Code"]})
    assert "UNRECOGNISED" in loud and "Weird Code" in loud
