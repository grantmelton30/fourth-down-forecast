from __future__ import annotations

import pandas as pd

from gate_artifact import REQUIRED_PROMOTION_GATES, write_gate_artifact
from model_identity import build_model_version
from sport import SportAdapter, SportProfile


class _Adapter(SportAdapter):
    def __init__(self, repo, cache, frame):
        super().__init__(SportProfile(key="nfl", label="NFL", repo=repo))
        self.cache = cache
        self.frame = frame

    def _cache_dir(self):
        return self.cache

    def backtest_frame(self):
        return self.frame


def _repo(tmp_path):
    repo = tmp_path / "nfl-model"
    config = repo / "config" / "nfl.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("revision: 1\n")
    (repo / "src").mkdir()
    (repo / "src" / "__init__.py").write_text("")
    (repo / "run_backtest.py").write_text("")
    return repo, config


def _frame():
    return pd.DataFrame({
        "game_id": ["g1"],
        "season": [2025],
        "week": [18],
        "kickoff": [pd.Timestamp("2025-12-31", tz="UTC")],
        "model_spread": [1.0],
        "market_spread": [0.5],
        "actual_margin": [3.0],
    })


def _version(repo, config, frame):
    return build_model_version("nfl", repo, config, frame)


def _passing_gates():
    return [
        {"name": name, "passed": True}
        for name in REQUIRED_PROMOTION_GATES["nfl"]
    ]


def test_adapter_allows_bets_only_from_matching_authoritative_artifact(tmp_path):
    repo, config = _repo(tmp_path)
    frame = _frame()
    cache = tmp_path / "cache"
    write_gate_artifact(
        cache,
        league="nfl",
        model_version=_version(repo, config, frame),
        data_cutoff="2025-12-31",
        gates=_passing_gates(),
        promotion_names=set(),
    )
    adapter = _Adapter(repo, cache, frame)
    assert adapter.bets_allowed() is True

    config.write_text("revision: 2\n")
    assert adapter.bets_allowed() is False
    assert "No valid gate artifact" in adapter.bet_block_reason()


def test_missing_promotion_measurement_blocks_adapter(tmp_path):
    repo, config = _repo(tmp_path)
    frame = _frame()
    cache = tmp_path / "cache"
    write_gate_artifact(
        cache,
        league="nfl",
        model_version=_version(repo, config, frame),
        data_cutoff="2025-12-31",
        gates=[{"name": "GATE_BLEND", "passed": True}],
        promotion_names={"GATE_BLEND", "GATE_RMSE_SPREAD"},
    )
    adapter = _Adapter(repo, cache, frame)
    assert adapter.bets_allowed() is False
    assert "GATE_RMSE_SPREAD" in adapter.bet_block_reason()
