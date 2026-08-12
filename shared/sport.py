"""Sport adapters -- the seam that lets one viewer and one exporter serve both repos.

`view.py` and `export.py` know nothing about nflverse, CFBD, ridge internals, or which
league they are looking at. They ask an adapter for ratings, games, a simulated
distribution, and the gate status; the adapter knows where its repo keeps those things.

Adding a sport means writing one adapter, not touching the viewer.

THE BET GATE LIVES HERE, deliberately. `bets_allowed()` is computed from the blend
artifact rather than passed in by a caller, so no UI path and no export path can decide on
its own to show picks. A model that has not demonstrated an edge shows numbers, never
selections.
"""

from __future__ import annotations

import json
import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from gate_artifact import FILENAME as GATE_FILENAME
from gate_artifact import GateArtifactError, load_gate_artifact
from model_identity import build_model_version

SHARED_DIR = Path(__file__).resolve().parent
WORKSPACE = SHARED_DIR.parent


def _load_model_package(repo: Path, key: str) -> str:
    """Load each repo's ``src`` under a unique name to prevent cross-league imports."""
    package = f"football_model_{key}_src"
    if package in sys.modules:
        return package
    init = Path(repo) / "src" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        package, init, submodule_search_locations=[str(init.parent)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load model package at {init}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    return package

BETS_BLOCKED_REASON = (
    "GATE_BLEND_INFORMATIVE has not passed for this sport. No component has "
    "demonstrated information beyond the market, so no picks are emitted. "
    "A model that declines to bet a market it has no edge in is working correctly."
)

# Minimum |t| on a blend coefficient before any pick may be shown.
T_THRESHOLD = 2.0
# A model less accurate than the line it bets against is never promoted, however
# suggestive its blend t-statistic looks. See SportAdapter.bets_allowed.
RMSE_MAX_RATIO = 1.0


def _select_default_period(
    games: pd.DataFrame, line_column: str, now: "object | None" = None
) -> "tuple[int, int] | None":
    """Choose the next priced slate, or the latest completed one.

    Sorting by season/week and taking the last row selected the farthest partially priced
    future week. Calendar proximity is the useful definition of "current" for a viewer.
    """
    required = {"season", "week", "kickoff", line_column}
    if games is None or games.empty or not required.issubset(games.columns):
        return None
    priced = games[games[line_column].notna()].copy()
    priced["_kickoff"] = pd.to_datetime(priced["kickoff"], utc=True, errors="coerce")
    priced = priced[priced["_kickoff"].notna()]
    if priced.empty:
        return None
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    current = (
        current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    )
    upcoming = priced[priced["_kickoff"] >= current]
    row = (
        upcoming.sort_values("_kickoff").iloc[0]
        if not upcoming.empty
        else priced.sort_values("_kickoff").iloc[-1]
    )
    return int(row["season"]), int(row["week"])


@dataclass(frozen=True)
class SportProfile:
    """Everything the viewer needs that differs between leagues."""

    key: str                      # "nfl" | "ncaa"
    label: str
    repo: Path
    key_numbers: tuple = (3, 7, 6, 10, 14, 4)
    rating_units: str = "EPA/play"
    has_conferences: bool = False
    market_line_basis: str = "closing"     # NCAA grades against the opener
    notes: str = ""


# --------------------------------------------------------------------------------------
# Base adapter
# --------------------------------------------------------------------------------------

@dataclass
class SportAdapter:
    """Contract the viewer codes against. Subclasses fill in the repo-specific bits."""

    profile: SportProfile
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def available(self) -> bool:
        """Whether this repo has been built far enough to show anything."""
        return self.profile.repo.exists() and (self.profile.repo / "src").exists()

    def status(self) -> dict:
        raise NotImplementedError

    def ratings(self, season: int, week: int) -> pd.DataFrame:
        raise NotImplementedError

    def rating_periods(self) -> pd.DataFrame:
        """Available (season, week) pairs, most recent first."""
        raise NotImplementedError

    def default_period(self):
        """The (season, week) the viewer should open on.

        Not simply the newest period: during the offseason the newest rating period is a
        scheduled-but-unplayed week with no market lines, which makes the Game explorer
        look broken on first load. Adapters override this to point at the most recent week
        that actually has priced games.
        """
        periods = self.rating_periods()
        if periods is None or periods.empty:
            return None
        row = periods.iloc[0]
        return int(row["season"]), int(row["week"])

    def games(self, season: int, week: int) -> pd.DataFrame:
        raise NotImplementedError

    def backtest_frame(self):
        raise NotImplementedError

    def simulate(self, game_id: str):
        """Return an object exposing margins / totals arrays, or None if unavailable."""
        raise NotImplementedError

    def _cache_dir(self) -> Path:
        raise NotImplementedError

    def _module(self, name: str):
        return importlib.import_module(f"{self._package}.{name}")

    def gate_table(self) -> pd.DataFrame:
        artifact = self.gate_artifact()
        if artifact is None:
            return pd.DataFrame(columns=["gate", "status", "observed", "detail"])
        rows = []
        for gate in artifact["gates"]:
            passed = gate.get("passed")
            rows.append({
                "gate": gate["name"],
                "status": "PASS" if passed is True else (
                    "FAIL" if passed is False else "NOT EVALUABLE"
                ),
                "observed": gate.get("observed", ""),
                "detail": gate.get("detail", ""),
            })
        return pd.DataFrame(rows)

    def gate_artifact(self) -> "dict | None":
        """Authoritative decision from the last backtest, or None on any ambiguity."""
        try:
            config_path = self.profile.repo / "config" / f"{self.profile.key}.yaml"
            frame = self.backtest_frame()
            if frame is None or frame.empty:
                return None
            model_version = build_model_version(
                self.profile.key, self.profile.repo, config_path, frame
            )
            payload = load_gate_artifact(
                self._cache_dir() / GATE_FILENAME,
                expected_league=self.profile.key,
                expected_model_version=model_version,
            )
            date_column = next(
                (c for c in ("kickoff", "gameday", "game_date", "date") if c in frame),
                None,
            )
            if date_column is None or not payload.get("data_cutoff"):
                return None
            current = pd.to_datetime(frame[date_column], utc=True, errors="coerce").max()
            recorded = pd.to_datetime(payload["data_cutoff"], utc=True, errors="coerce")
            if pd.isna(current) or pd.isna(recorded) or current.date() != recorded.date():
                return None
            return payload
        except (GateArtifactError, OSError, ValueError):
            return None

    def blend_weights(self) -> dict:
        path = self._cache_dir() / "blend_weights.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except Exception:  # noqa: BLE001 - a corrupt artifact must not open the gate
            return {}

    # -- the gate -----------------------------------------------------------------------

    # Candidate column names, because the two repos disagree: nfl-model grades against
    # market_spread / market_total, ncaa-model against the opener it was built to test.
    _MARKET_COLS = {
        "spread": ("market_spread", "spread_open", "spread_close"),
        "total": ("market_total", "total_open", "total_close"),
    }
    _MODEL_COLS = {"spread": "model_spread", "total": "model_total"}
    _ACTUAL_COLS = {"spread": "actual_margin", "total": "actual_total"}

    def rmse_vs_market(self) -> dict:
        """Model RMSE divided by the market's, per market. Above 1.0 means the model is
        LESS accurate than the number it would be betting against."""
        import numpy as np

        frame = self.backtest_frame()
        if frame is None or not len(frame):
            return {}
        out = {}
        for market, model_col in self._MODEL_COLS.items():
            actual_col = self._ACTUAL_COLS[market]
            market_col = next(
                (c for c in self._MARKET_COLS[market] if c in frame.columns), None
            )
            if market_col is None or model_col not in frame or actual_col not in frame:
                continue
            d = frame[[model_col, market_col, actual_col]].dropna()
            if len(d) < 100:
                continue
            a = d[actual_col].to_numpy(float)
            mr = float(np.sqrt(np.mean((d[model_col].to_numpy(float) - a) ** 2)))
            kr = float(np.sqrt(np.mean((d[market_col].to_numpy(float) - a) ** 2)))
            if kr > 0:
                out[market] = mr / kr
        return out

    def bets_allowed(self) -> bool:
        """Read the sole promotion decision produced by the backtest; fail closed."""
        artifact = self.gate_artifact()
        return bool(artifact and artifact.get("bets_allowed") is True)

    def bet_block_reason(self) -> str:
        artifact = self.gate_artifact()
        if artifact is None:
            return (
                "No valid gate artifact was found. Run the backtest for this exact model "
                "and data version; missing, stale, or malformed evidence always blocks picks."
            )
        blockers = ", ".join(artifact.get("blockers") or ["unspecified gate"])
        return f"Picks are blocked by: {blockers}."


# --------------------------------------------------------------------------------------
# NFL
# --------------------------------------------------------------------------------------

class NFLAdapter(SportAdapter):
    """Reads the nfl-model repo.

    Its bet gate fails by design -- see APPENDIX A of NFL_PLAYBOOK.md -- so
    `bets_allowed()` returns False and the exporter writes an explanatory sheet instead of
    picks.
    """

    def __init__(self, repo: Path):
        super().__init__(
            SportProfile(
                key="nfl",
                label="NFL",
                repo=repo,
                key_numbers=(3, 7, 6, 10, 14, 4),
                has_conferences=False,
                market_line_basis="closing",
                notes=(
                    "Graded against the CLOSING line -- nflverse publishes no opening "
                    "lines. Null result on spreads and totals; see APPENDIX A."
                ),
            )
        )
        self._package = _load_model_package(self.profile.repo, self.profile.key)

    # -- lazy machinery -----------------------------------------------------------------

    def _cfg(self):
        if "cfg" not in self._cache:
            self._cache["cfg"] = self._module("config").load_config()
        return self._cache["cfg"]

    def _cache_dir(self) -> Path:
        return Path(self._module("config").CACHE_DIR)

    def _schedules(self) -> pd.DataFrame:
        if "sched" not in self._cache:
            ingest = self._module("ingest")
            self._cache["sched"] = ingest.load_schedules(self._cfg().train_seasons)
        return self._cache["sched"]

    def _walkforward(self) -> pd.DataFrame:
        if "wf" not in self._cache:
            path = self._wf_path()
            self._cache["wf"] = (
                pd.read_parquet(path)
                if path.exists()
                else pd.DataFrame(columns=["season", "week", "team"])
            )
        return self._cache["wf"]

    def _wf_path(self) -> Path:
        """Read a completed ratings artifact; opening the UI must never start a refit."""
        try:
            cfg = self._cfg()
            lo, ld = cfg.effective_lambdas
            seasons = cfg.train_seasons
            tag = f"default_o{lo:g}_d{ld:g}_p{cfg.pace.lambda_:g}"
            expected = self._cache_dir() / (
                f"walkforward_ratings_{tag}_{min(seasons)}_{max(seasons)}.parquet"
            )
            if expected.exists():
                return expected
        except Exception:  # noqa: BLE001 - fall back to any completed viewer artifact
            pass
        candidates = list(
            self._cache_dir().glob("walkforward_ratings_default*.parquet")
        )
        if not candidates:
            candidates = list(self._cache_dir().glob("walkforward_ratings_*.parquet"))
        return (
            max(candidates, key=lambda path: path.stat().st_mtime)
            if candidates
            else self._cache_dir() / "__missing__"
        )

    def _sim_machinery(self, season: int):
        """Drive model, endgame table and field-position distribution for a season."""
        key = f"mach_{season}"
        if key not in self._cache:
            ingest = self._module("ingest")
            drive_model = self._module("drive_model")
            drive_tools = self._module("drives")

            cfg = self._cfg()
            pbp = ingest.load_pbp(cfg.train_seasons)
            drives = drive_tools.build_drive_table(pbp, cache_key="main")
            wf = self._walkforward()
            self._cache[key] = (
                drive_model.fit_drive_model(drives, wf, cfg, as_of_season=season),
                drive_model.fit_endgame_table(drives, cfg, as_of_season=season),
                drive_tools.fit_start_field_position(drives),
            )
        return self._cache[key]

    def _net(self, off: pd.Series, dfn: pd.Series):
        """Team quality on one scale: its offense's value minus what its defense concedes.

        Higher def_rating means a worse defense, so it enters negatively here even though
        a *matchup* adds the opponent's def_rating. Keeping both conventions straight is
        why this lives in one place.
        """
        cfg = self._cfg()
        return (
            cfg.ratings.off_weight * off.to_numpy()
            - cfg.ratings.def_weight * dfn.to_numpy()
        )

    # -- contract ------------------------------------------------------------------------

    def status(self) -> dict:
        frame = self.backtest_frame()
        return {
            "repo": str(self.profile.repo),
            "built": self.available,
            "backtest_games": 0 if frame is None else len(frame),
            "bets_allowed": self.bets_allowed(),
            "market_basis": self.profile.market_line_basis,
        }

    def rating_periods(self) -> pd.DataFrame:
        wf = self._walkforward()
        return (
            wf[["season", "week"]]
            .drop_duplicates()
            .sort_values(["season", "week"], ascending=[False, False])
            .reset_index(drop=True)
        )

    def ratings(self, season: int, week: int) -> pd.DataFrame:
        wf = self._walkforward()
        cur = wf[(wf["season"] == season) & (wf["week"] == week)].set_index("team")
        if cur.empty:
            return pd.DataFrame()

        out = cur[["off_rating", "def_rating", "pace_rating", "n_games"]].copy()
        out["net_rating"] = self._net(out["off_rating"], out["def_rating"])
        out["off_rank"] = out["off_rating"].rank(ascending=False).astype(int)
        # Higher def_rating = worse defense, so the best defense ranks first ascending.
        out["def_rank"] = out["def_rating"].rank(ascending=True).astype(int)
        out["net_rank"] = out["net_rating"].rank(ascending=False).astype(int)
        out["pace_rank"] = out["pace_rating"].rank(ascending=False).astype(int)

        prev = self._previous_net(season, week)
        out["net_change"] = (
            out["net_rating"] - prev.reindex(out.index) if prev is not None else float("nan")
        )
        return out.sort_values("net_rating", ascending=False)

    def _previous_net(self, season: int, week: int):
        wf = self._walkforward()
        earlier = wf[
            (wf["season"] < season) | ((wf["season"] == season) & (wf["week"] < week))
        ]
        if earlier.empty:
            return None
        last = earlier.sort_values(["season", "week"]).iloc[-1]
        prev = wf[
            (wf["season"] == last["season"]) & (wf["week"] == last["week"])
        ].set_index("team")
        return pd.Series(
            self._net(prev["off_rating"], prev["def_rating"]), index=prev.index
        )

    def default_period(self):
        """Next upcoming priced regular-season slate, else the latest completed one."""
        s = self._schedules()
        if "game_type" in s.columns:
            s = s[s["game_type"].eq("REG")]
        return _select_default_period(s, "spread_line") or super().default_period()

    def games(self, season: int, week: int) -> pd.DataFrame:
        s = self._schedules()
        sel = s[(s["season"] == season) & (s["week"] == week)]
        cols = [
            "game_id", "kickoff", "away_team", "home_team", "spread_line", "total_line",
            "result", "total", "roof", "location", "away_rest", "home_rest", "temp",
            "wind",
        ]
        return sel[[c for c in cols if c in sel.columns]].reset_index(drop=True)

    def backtest_frame(self):
        # nfl-model tags this artifact with a hash of the config that produced it, so a
        # bare "backtest_frame.parquet" is no longer what gets written. Ask the repo for
        # the path; fall back to the newest tagged file if its config cannot be imported.
        path = None
        try:
            candidate = self._module("backtest").backtest_frame_path(self._cfg())
            if candidate.exists():
                path = candidate
        except Exception:  # noqa: BLE001 - repo not built far enough to import
            pass
        if path is None:
            cands = list(self._cache_dir().glob("backtest_frame*.parquet"))
            if not cands:
                return None
            path = max(cands, key=lambda p: p.stat().st_mtime)
        if "bt" not in self._cache:
            self._cache["bt"] = pd.read_parquet(path)
        return self._cache["bt"]

    def simulate(self, game_id: str):
        """Simulate one game with the same machinery the backtest used."""
        context = self._module("context")
        ingest = self._module("ingest")
        injuries = self._module("injuries")
        nfelo = self._module("nfelo")
        qb_module = self._module("qb")
        ratings = self._module("ratings")
        simulate = self._module("simulate")

        s = self._schedules()
        row = s[s["game_id"] == game_id]
        if row.empty:
            return None
        g = row.iloc[0]
        season, week = int(g["season"]), int(g["week"])
        cfg = self._cfg()
        wf = self._walkforward()
        try:
            rt = ratings.ratings_at(wf, season, week)
        except KeyError:
            return None
        if g["home_team"] not in rt.index or g["away_team"] not in rt.index:
            return None

        model, endgame, start_fp = self._sim_machinery(season)
        venue = context.estimate_venue_hfa(s, wf, cfg, before_season=season)
        if "injury_burden" not in self._cache:
            report = ingest.load_injuries(cfg.train_seasons)
            snaps = ingest.load_snap_counts(cfg.train_seasons)
            self._cache["injury_burden"] = injuries.burden_lookup(
                injuries.build_injury_burden(report, snaps))
        ctx = context.build_context(
            g, cfg, venue_hfa=venue, allow_network=False,
            resting_starters=context.load_resting_starters(),
            injury_burden=self._cache["injury_burden"],
        )
        if "qb_adjustments" not in self._cache:
            self._cache["qb_adjustments"] = nfelo.qb_adjustments_by_game(
                nfelo.load_qb_elos(), s)
        qb = qb_module.qb_points_for_game(
            game_id, self._cache["qb_adjustments"], cfg)
        ctx = ctx.with_qb(qb.points, qb.note)
        return simulate.simulate_game(
            g["home_team"], g["away_team"], rt, model, ctx, cfg, start_fp,
            endgame=endgame, neutral_site=g.get("location") == "Neutral",
        )

    def gate_table(self) -> pd.DataFrame:
        return super().gate_table()


# --------------------------------------------------------------------------------------
# NCAA
# --------------------------------------------------------------------------------------

class NCAAAdapter(SportAdapter):
    """Reads the ncaa-model repo.

    Per the amended build order (ingest -> ratings -> viewer -> simulator -> ...), this
    adapter is expected to be partially functional: ratings work as soon as `ratings.py`
    lands, and the Game explorer stays dark until the simulator exists. Every method
    degrades to an empty result rather than raising, so the viewer renders whatever is
    ready.
    """

    def __init__(self, repo: Path):
        super().__init__(
            SportProfile(
                key="ncaa",
                label="NCAA",
                repo=repo,
                key_numbers=(3, 7, 10, 14, 17, 21),
                has_conferences=True,
                market_line_basis="opening",
                notes=(
                    "Graded against the OPENING line (CFBD spread_open), with the close "
                    "retained for CLV. This is the test the NFL build could not run."
                ),
            )
        )
        self._package = _load_model_package(self.profile.repo, self.profile.key)

    def _cache_dir(self) -> Path:
        """Honour NCAA_MODEL_CACHE_DIR the way the repo's own modules do.

        Globbing `repo/data/cache` unconditionally was a real bug: the build runs with the
        cache pointed off the iCloud-synced Desktop, so the viewer found zero rating
        periods and rendered an empty Ratings tab against a fully built repo.
        """
        try:
            return Path(self._module("config").CACHE_DIR)
        except Exception:  # noqa: BLE001 - repo not built far enough yet
            import os

            return Path(
                os.environ.get(
                    "NCAA_MODEL_CACHE_DIR", self.profile.repo / "data" / "cache"
                )
            ).expanduser()

    def status(self) -> dict:
        frame = self.backtest_frame()
        return {
            "repo": str(self.profile.repo),
            "built": self.available,
            "backtest_games": 0 if frame is None else len(frame),
            "bets_allowed": self.bets_allowed(),
            "market_basis": self.profile.market_line_basis,
        }

    def _wf_path(self) -> Path:
        """Prefer the shipped artifact over the sweep variants, which are hyperparameter
        experiments rather than the ratings the model actually uses."""
        try:
            tagged = self._module("ratings").walkforward_path(self._cfg())
            if tagged.exists():
                return tagged
        except Exception:  # noqa: BLE001 - repo not built far enough to import
            pass
        # Fall back to the newest artifact on disk. Prefer "default" keys over "sweep"
        # ones, which are hyperparameter experiments rather than the shipped ratings.
        cands = [
            p for p in self._cache_dir().glob("walkforward_ratings_default*.parquet")
        ] or sorted(self._cache_dir().glob("walkforward_ratings_*.parquet"))
        if not cands:
            return self._cache_dir() / "__missing__"
        return max(cands, key=lambda p: p.stat().st_mtime)

    def rating_periods(self) -> pd.DataFrame:
        p = self._wf_path()
        if not p.exists():
            return pd.DataFrame(columns=["season", "week"])
        wf = pd.read_parquet(p)
        return (
            wf[["season", "week"]].drop_duplicates()
            .sort_values(["season", "week"], ascending=[False, False])
            .reset_index(drop=True)
        )

    def ratings(self, season: int, week: int) -> pd.DataFrame:
        p = self._wf_path()
        if not p.exists():
            return pd.DataFrame()
        wf = pd.read_parquet(p)
        cur = wf[(wf["season"] == season) & (wf["week"] == week)]
        if cur.empty:
            return pd.DataFrame()
        out = cur.set_index("team")
        keep = [
            c for c in ("off_rating", "def_rating", "pace_rating", "conference", "n_games")
            if c in out.columns
        ]
        out = out[keep].copy()
        if {"off_rating", "def_rating"} <= set(out.columns):
            # Team quality: its offense minus what its defense concedes. Higher
            # def_rating means a worse defense, so it enters negatively here even though a
            # *matchup* adds the opponent's def_rating.
            out["net_rating"] = out["off_rating"] - out["def_rating"]
            out["off_rank"] = out["off_rating"].rank(ascending=False).astype(int)
            out["def_rank"] = out["def_rating"].rank(ascending=True).astype(int)
            out["net_rank"] = out["net_rating"].rank(ascending=False).astype(int)
            out = out.sort_values("net_rating", ascending=False)
        return out

    def default_period(self):
        """Next upcoming priced slate, else the latest completed one."""
        m = self._market()
        return _select_default_period(m, "spread_open") or super().default_period()

    def _market(self):
        path = self._cache_dir() / "market.parquet"
        if "market" not in self._cache:
            self._cache["market"] = (
                pd.read_parquet(path) if path.exists() else pd.DataFrame()
            )
        return self._cache["market"]

    def games(self, season: int, week: int) -> pd.DataFrame:
        """The slate, with both the opener (what we grade against) and the close."""
        m = self._market()
        if m is None or m.empty:
            return pd.DataFrame()
        sel = m[(m["season"] == season) & (m["week"] == week)].copy()
        if sel.empty:
            return pd.DataFrame()
        sel = sel.rename(columns={
            "spread_open": "spread_line", "total_open": "total_line",
            "actual_margin": "result", "actual_total": "total",
        })
        cols = [
            "game_id", "kickoff", "away_team", "home_team", "spread_line", "total_line",
            "spread_close", "total_close", "result", "total", "restricted",
            "cross_tier", "provider",
        ]
        out = sel[[c for c in cols if c in sel.columns]].reset_index(drop=True)
        # Priced games first. A college week contains a long tail of FCS-vs-FCS games with
        # no line at all, and they sort to the top alphabetically -- so the Game explorer
        # opened on a matchup with nothing to compare the model against, which is the whole
        # point of the tab.
        if "spread_line" in out.columns:
            out["_priced"] = out["spread_line"].notna()
            sort_by = ["_priced"] + (["kickoff"] if "kickoff" in out.columns else [])
            out = (out.sort_values(sort_by, ascending=[False] + [True] * (len(sort_by) - 1))
                      .drop(columns="_priced").reset_index(drop=True))
        return out

    def backtest_frame(self):
        # ncaa-model writes backtest_frame_{cache_key}.parquet -- "default" for the
        # shipped run and "sweep{i}" for the pre-registered grid. This looked for a bare
        # backtest_frame.parquet, which the repo has never written, so it returned None
        # and every consumer (Diagnostics, the accuracy precondition) silently saw nothing.
        for name in ("backtest_frame_default.parquet", "backtest_frame.parquet"):
            path = self._cache_dir() / name
            if path.exists():
                return pd.read_parquet(path)
        cands = sorted(self._cache_dir().glob("backtest_frame_*.parquet"))
        return pd.read_parquet(cands[-1]) if cands else None

    def _cfg(self):
        if "cfg" not in self._cache:
            self._cache["cfg"] = self._module("config").load_config()
        return self._cache["cfg"]

    def _walkforward(self) -> pd.DataFrame:
        if "wf" not in self._cache:
            p = self._wf_path()
            self._cache["wf"] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        return self._cache["wf"]

    def _games_with_venues(self) -> pd.DataFrame:
        """The full schedule joined to venue coordinates.

        `games()` returns the market slate, which carries no venue -- and context needs
        lat/lon/dome to resolve weather and venue_id to resolve home-field. Costs at most
        one CFBD call for /venues; everything else is served from the permanent cache.
        """
        if "gv" not in self._cache:
            ingest, venues = self._module("ingest"), self._module("venues")
            BudgetedCFBD = self._module("cfbd_client").BudgetedCFBD

            cfg = self._cfg()
            client = BudgetedCFBD(cfg)
            seasons = list(range(cfg.seasons.train_start, cfg.seasons.current))
            g = ingest.load_games(client, seasons)
            self._cache["gv"] = venues.attach_venues(g, venues.load_venues(client))
        return self._cache["gv"]

    def _drive_table(self) -> pd.DataFrame:
        """The shared-schema drive table, built from the permanent JSON cache."""
        if "drives" not in self._cache:
            import json
            CACHE_DIR = self._module("config").CACHE_DIR
            drive_tools = self._module("drives")
            DRIVE_COLUMNS = drive_tools.DRIVE_COLUMNS

            cache = Path(CACHE_DIR) / "drive_table.parquet"
            if cache.exists():
                df = pd.read_parquet(cache)
                if not set(DRIVE_COLUMNS) - set(df.columns):
                    self._cache["drives"] = df
                    return df
            frames = []
            for path in sorted(Path(CACHE_DIR).glob("drives_[0-9][0-9][0-9][0-9].json")):
                d = pd.json_normalize(json.loads(path.read_text()))
                d["season"] = int(path.stem.split("_")[1])
                frames.append(d)
            if not frames:
                self._cache["drives"] = pd.DataFrame()
                return self._cache["drives"]
            raw = pd.concat(frames, ignore_index=True).rename(
                columns={"gameId": "game_id"}
            )
            gv = self._games_with_venues()
            self._cache["drives"] = drive_tools.build_drive_table(
                raw, gv[["game_id", "week"]], cache_path=cache
            )
        return self._cache["drives"]

    def _sim_machinery(self, season: int):
        """Drive model, endgame table and field-position distribution for a season.

        Fit once per season and held on the adapter, which `view.get_adapter` keeps behind
        st.cache_resource -- the multinomial is fit on ~120k drives and is far too slow to
        redo on every interaction.
        """
        key = f"mach_{season}"
        if key not in self._cache:
            drive_model = self._module("drive_model")
            drive_tools = self._module("drives")

            cfg = self._cfg()
            drives = self._drive_table()
            if drives.empty:
                return None
            earlier = drives[drives["season"] < season]
            self._cache[key] = (
                drive_model.fit_drive_model(
                    drives, self._walkforward(), cfg, as_of_season=season
                ),
                drive_model.fit_endgame_table(drives, cfg, as_of_season=season),
                # Field position is stationary enough that the earliest season, which has
                # no prior, falls back to the whole table rather than failing.
                drive_tools.fit_start_field_position(
                    earlier if not earlier.empty else drives
                ),
            )
        return self._cache[key]

    def simulate(self, game_id: str):
        """Simulate one game with the machinery the projection script uses.

        Returns None rather than raising for any missing piece: the viewer treats None as
        "not available for this game" and keeps rendering the rest of the page.
        """
        context = self._module("context")
        ratings = self._module("ratings")
        simulate = self._module("simulate")

        gv = self._games_with_venues()
        row = gv[gv["game_id"].astype(str) == str(game_id)]
        if row.empty:
            return None
        g = row.iloc[0]
        season, week = int(g["season"]), int(g["week"])
        wf = self._walkforward()
        if wf.empty:
            return None
        try:
            rt = ratings.ratings_at(wf, season, week)
        except KeyError:
            return None

        machinery = self._sim_machinery(season)
        if machinery is None:
            return None
        model, endgame, start_fp = machinery

        cfg = self._cfg()
        venue = context.estimate_venue_hfa(
            gv, cfg, walkforward=wf, as_of=g.get("kickoff")
        )
        # allow_network=False: a live Open-Meteo call per game would make the tab feel
        # broken. Readings already in weather.parquet are still used.
        ctx = context.build_context(g, cfg, venue_hfa=venue, allow_network=False)
        return simulate.simulate_game(
            g["homeTeam"], g["awayTeam"], rt, model, cfg, start_fp,
            context_adj=ctx, endgame=endgame,
            neutral_site=bool(g.get("neutralSite", False)),
        )

    def gate_table(self) -> pd.DataFrame:
        return super().gate_table()


# --------------------------------------------------------------------------------------

_REGISTRY = {
    "nfl": (NFLAdapter, "nfl-model"),
    "ncaa": (NCAAAdapter, "ncaa-model"),
}


def load_adapter(sport: str, workspace: "Path | None" = None) -> SportAdapter:
    """`sport` is the config argument -- "nfl" or "ncaa"."""
    key = (sport or "nfl").strip().lower()
    if key not in _REGISTRY:
        raise ValueError(f"unknown sport {sport!r}; expected one of {sorted(_REGISTRY)}")
    cls, dirname = _REGISTRY[key]
    root = Path(workspace) if workspace else WORKSPACE
    return cls(root / dirname)


def available_sports(workspace: "Path | None" = None) -> list:
    root = Path(workspace) if workspace else WORKSPACE
    return [k for k, (_, d) in _REGISTRY.items() if (root / d).exists()]
