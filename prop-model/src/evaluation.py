"""Chronological eligibility and holdout boundaries shared by prop evaluations."""
from __future__ import annotations

import pandas as pd


def eligible_rows(frame, denominator, minimum):
    ordered = frame.sort_values("kickoff").copy()
    if ordered.duplicated(["player_id", "kickoff"]).any():
        raise ValueError("duplicate player/kickoff observations")
    prior = ordered.groupby("player_id")[denominator].transform(
        lambda x: x.shift(1).expanding().mean())
    return ordered[prior >= minimum].copy()


def chronological_split(frame, holdout):
    train = frame[frame.season < holdout].copy()
    test = frame[frame.season == holdout].copy()
    if train.empty or test.empty:
        raise ValueError("evaluation requires earlier training seasons and a populated holdout")
    return train, test
