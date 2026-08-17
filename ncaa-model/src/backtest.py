"""L1 mean layer, walk-forward backtest, and the gates.

THE SIMULATOR IS NOT IN THE MEAN PATH (PATCH 04 §1.3, DECISIONS D4). `E[margin]` and
`E[total]` come from a direct linear projection off the ridge ratings. On the NFL build the
drive simulator contributed every one of the 5.02 points of excess dispersion in the mean
projection while scoring worse than the direct projection off the same ratings, so it is
not used here to produce a mean at all. (5.02 is a completed historical measurement cited
from NFL_PLAYBOOK.md Appendix A, not a live metric.)

Everything is graded against the OPENER. The close is carried alongside solely to measure
CLV: whether the market moved toward the model after the model committed.
"""

from __future__ import annotations

from dataclasses import asdict,dataclass,replace
from pathlib import Path

import numpy as np
import pandas as pd

from ._shared import robust_inference, validated_model
from .config import CACHE_DIR,Config,build_cache_signature,frame_signature,read_cached_frame,write_cached_frame
from .context import build_context, estimate_venue_hfa
from .drive_model import fit_drive_model, fit_endgame_table
from .drives import build_drive_table, fit_start_field_position
from .ratings import net_epa_vec, ratings_at
from .simulate import simulate_game

FIRST_GRADED_WEEK = 4


@dataclass
class GateResult:
    name: str
    passed: bool
    observed: str
    detail: str = ""

    def __str__(self) -> str:
        return f"  [{'PASS' if self.passed else 'FAIL'}] {self.name:<26} {self.observed}"


# --------------------------------------------------------------------------------------
# L1 -- direct linear projection off the ridge ratings
# --------------------------------------------------------------------------------------

def build_features(
    market: pd.DataFrame, walkforward: pd.DataFrame, cfg: Config,
    challenger_features: "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    """Attach each game's as-of ratings and form the linear predictors.

    `net_diff` drives the spread; `pace_sum` and `eff_sum` drive the total. College pace
    varies far more than NFL pace, which is why the totals path gets its own predictor
    rather than being a by-product of the spread.
    """
    wf = walkforward[[
        "season", "week", "team", "off_rating", "def_rating", "pace_rating"
    ]]
    h = wf.rename(columns={
        "team": "home_team", "off_rating": "h_off", "def_rating": "h_def",
        "pace_rating": "h_pace"})
    a = wf.rename(columns={
        "team": "away_team", "off_rating": "a_off", "def_rating": "a_def",
        "pace_rating": "a_pace"})

    df = market.merge(h, on=["season", "week", "home_team"], how="inner")
    df = df.merge(a, on=["season", "week", "away_team"], how="inner")

    # Higher def_rating = worse defense, so facing it RAISES the opposing offense. The
    # combination lives in ratings.net_epa_vec so that this linear projection and the drive
    # simulator cannot disagree about what "net efficiency" means.
    net_home = net_epa_vec(df["h_off"], df["a_def"], cfg)
    net_away = net_epa_vec(df["a_off"], df["h_def"], cfg)
    df["net_diff"] = net_home - net_away
    df["eff_sum"] = net_home + net_away
    df["pace_sum"] = df["h_pace"] + df["a_pace"]
    df["is_home"] = np.where(df["neutralSite"].fillna(False), 0.0, 1.0)
    if challenger_features is not None and len(challenger_features):
        additions = challenger_features.drop_duplicates("game_id")
        df = df.merge(additions, on="game_id", how="left", suffixes=("", "_challenger"))
    return df


def _validated_challenger(train: pd.DataFrame, target: str, base: list[str],
                          suffix: str):
    candidates = [c for c in train if c.endswith(suffix)]
    if train["season"].nunique() < 3 or not candidates:
        return None
    try:
        return validated_model.fit_validated_ridge(
            train, base_features=base, candidate_features=candidates, target=target,
            alpha=45.0, minimum_fit_rows=500, minimum_improvement=0.05,
        )
    except ValueError:
        return None


def project_walkforward(feats: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Fit the ratings->points map on strictly earlier seasons, apply to the current one.

    An expanding window: projecting 2023 uses only 2019-2022 to learn the scale. The map is
    deliberately tiny -- two predictors for the spread, two for the total -- because with
    ~800 FBS games a season anything richer backtests well and loses money.
    """
    out = []
    for season in sorted(feats["season"].unique()):
        train = feats[(feats["season"] < season) & feats["actual_margin"].notna()]
        test = feats[feats["season"] == season].copy()
        if len(train) < 300:
            test["model_spread"] = np.nan
            test["model_total"] = np.nan
            test["model_spread_pure"] = np.nan
            test["model_total_pure"] = np.nan
            test["model_spread_market_adjusted"] = np.nan
            test["model_total_market_adjusted"] = np.nan
            out.append(test)
            continue

        Xs = np.column_stack([np.ones(len(train)), train["net_diff"], train["is_home"]])
        bs = np.linalg.lstsq(Xs, train["actual_margin"].to_numpy(float), rcond=None)[0]
        test["model_spread"] = bs[0] + bs[1] * test["net_diff"] + bs[2] * test["is_home"]

        Xt = np.column_stack([np.ones(len(train)), train["pace_sum"], train["eff_sum"]])
        bt = np.linalg.lstsq(Xt, train["actual_total"].to_numpy(float), rcond=None)[0]
        test["model_total"] = bt[0] + bt[1] * test["pace_sum"] + bt[2] * test["eff_sum"]

        spread_challenger = _validated_challenger(
            train, "actual_margin", ["net_diff", "is_home"], "_diff")
        total_challenger = _validated_challenger(
            train, "actual_total", ["pace_sum", "eff_sum"], "_sum")
        test["feature_spread_promoted"] = bool(
            spread_challenger and spread_challenger.challenger_promoted)
        test["feature_total_promoted"] = bool(
            total_challenger and total_challenger.challenger_promoted)
        if test["feature_spread_promoted"].iloc[0]:
            test["model_spread"] = spread_challenger.predict(test)
        if test["feature_total_promoted"].iloc[0]:
            test["model_total"] = total_challenger.predict(test)

        # THE PURE PROJECTION (Phase 0E). Everything below this line touches a betting
        # line; these two columns are the last point at which the projection is a function
        # of features and nothing else. They are persisted because the shipped
        # `model_total` is opener-derived in three separate places (see DECISIONS
        # "opener-injection points"), which means no gate, blend or CLV figure in this
        # repo has ever been computed on an uncontaminated projection. Any future claim of
        # edge must be measured on `*_pure` against the CLOSE, or it is measuring the
        # anchor again.
        test["model_spread_pure"] = test["model_spread"]
        test["model_total_pure"] = test["model_total"]

        # GAIN CORRECTION (GATE_SCALE). Least squares is an RMSE-minimizing estimator, so
        # its fitted values are attenuated by construction: sd(pred) = R * sd(y). That is
        # optimal for RMSE and wrong for betting, because a compressed projection produces
        # compressed *disagreements*, and a fixed points threshold then silently skips
        # real edges. The raw totals projection came out at a 0.787 SD ratio -- 21% too
        # small -- which under-flags the slate rather than over-flagging it.
        #
        # The gain is fit on strictly earlier seasons only, so it carries no lookahead,
        # and it is applied about the projection's own mean so the level (and therefore
        # the intercept `a` that GATE_UNBIASED watches) is preserved.
        for kind, col, mkt in (
            ("spread", "model_spread", "spread_open"),
            ("total", "model_total", "total_open"),
        ):
            g = _fit_gain(train, col, mkt, kind)
            test[f"gain_{kind}"] = g

            # TIME-VARYING INTERCEPT. The projection carries a level bias that grows
            # monotonically within a season -- +0.62 points at week 4, +1.74 by week 13.
            # A full-sample GATE_UNBIASED cannot see it: the average of a drift is small
            # (+0.219) while the drift itself is large enough that, past week 12, the
            # constant offset alone exceeds the selection threshold. Every game then
            # clears on the same side and the picks carry no game-specific information at
            # all -- which is exactly what 7/7 OVER looked like.
            #
            # Mechanism (measured, not assumed): pace ratings do not inflate, shrinkage
            # does not relax, and the garbage filter is stable. `eff_sum` drifts by
            # ~0.003 PPA/play across a season, and the fitted totals coefficient on it is
            # large enough to turn that into most of a point.
            train_pred = _project(train, bs if kind == "spread" else bt, kind)
            train_center=float(train_pred.mean()); train_adjusted=train_center+(train_pred-train_center)*g
            bias = _fit_week_bias(train, train_adjusted, mkt)
            test[f"week_bias_{kind}"] = test["week"].map(bias).fillna(0.0)
            test[f"model_{kind}_market_adjusted"] = train_center+(test[col]-train_center)*g-test[f"week_bias_{kind}"]
        out.append(test)
    return pd.concat(out, ignore_index=True)


def _project(rows: pd.DataFrame, beta: np.ndarray, kind: str) -> pd.Series:
    """Apply a fitted projection to arbitrary rows, so the training-season bias can be
    measured on the same quantity that is applied to the test season."""
    if kind == "spread":
        vals = beta[0] + beta[1] * rows["net_diff"] + beta[2] * rows["is_home"]
    else:
        vals = beta[0] + beta[1] * rows["pace_sum"] + beta[2] * rows["eff_sum"]
    return pd.Series(vals, index=rows.index)


def _fit_week_bias(
    train: pd.DataFrame, pred: pd.Series, mkt: str, shrink: float = 40.0
) -> dict:
    """Per-week level bias, estimated on earlier seasons only.

    Shrunk toward the global bias by `n / (n + shrink)` so thin weeks -- 15 and 16 carry
    a handful of games -- cannot contribute a wild correction. Subtracting an unshrunk
    week-16 estimate built on two games would replace one defect with another.
    """
    ok = pred.notna() & train[mkt].notna()
    if int(ok.sum()) < 300:
        return {}
    resid = (pred[ok] - train.loc[ok, mkt])
    global_bias = float(resid.mean())
    grp = resid.groupby(train.loc[ok, "week"])
    n, mean = grp.size(), grp.mean()
    return ((n * mean + shrink * global_bias) / (n + shrink)).to_dict()


def _fit_gain(train: pd.DataFrame, col: str, mkt: str, kind: str) -> float:
    """Scalar gain matching the projection's dispersion to the market's, from past data.

    Estimated on the training seasons -- where the projection is recomputed on the same
    expanding-window basis -- never on the season being predicted.
    """
    ref = train.dropna(subset=[mkt])
    if len(ref) < 200:
        return 1.0
    if kind == "spread":
        X = np.column_stack([np.ones(len(ref)), ref["net_diff"], ref["is_home"]])
        y = ref["actual_margin"].to_numpy(float)
    else:
        X = np.column_stack([np.ones(len(ref)), ref["pace_sum"], ref["eff_sum"]])
        y = ref["actual_total"].to_numpy(float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    pred_sd = float(np.std(X @ beta))
    mkt_sd = float(ref[mkt].std())
    if pred_sd <= 1e-9 or mkt_sd <= 1e-9:
        return 1.0
    # Cap the correction: a gain far from 1 means the projection is not merely attenuated,
    # and inflating it would amplify noise rather than restore signal.
    return float(np.clip(mkt_sd / pred_sd, 0.5, 2.0))


def walk_forward(
    cfg: Config,
    market: pd.DataFrame,
    walkforward: pd.DataFrame,
    cache_key: "str | None" = "default",
    challenger_features: "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    """One row per gradeable game, with model projections attached."""
    path = CACHE_DIR / f"backtest_frame_{cache_key}.parquet"
    signature=build_cache_signature(builder=Path(__file__),config=asdict(cfg),inputs={
        "market":frame_signature(market,["game_id","kickoff","completed","spread_open","spread_close","total_open","total_close","actual_margin","actual_total"]),
        "walkforward":frame_signature(walkforward,["season","week","as_of","team","off_rating","def_rating","pace_rating"])},artifact_version=2)
    if cache_key:
        cached=read_cached_frame(path,["game_id","model_spread","model_total","model_spread_pure","model_total_pure"],signature)
        if cached is not None: return cached

    graded = market[
        market["completed"].fillna(False)
        & (market["week"] >= FIRST_GRADED_WEEK)
        & market["actual_margin"].notna()
    ].copy()
    frame = project_walkforward(
        build_features(graded, walkforward, cfg, challenger_features), cfg)
    if cache_key:
        write_cached_frame(frame,path,signature)
    return frame


def select_window(frame: pd.DataFrame, seasons: list, restricted: bool = True):
    sub = frame[frame["season"].isin(seasons) & frame["has_opener"]]
    if restricted:
        sub = sub[sub["restricted"]]
    return sub.dropna(subset=["model_spread", "model_total"])


# --------------------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------------------

def gate_opener_coverage(market: pd.DataFrame, cfg: Config, seasons: list) -> GateResult:
    sub = market[market["restricted"] & market["season"].isin(seasons)]
    cov = float(sub["has_opener"].mean()) if len(sub) else 0.0
    sep = (sub["spread_open"] - sub["spread_close"]).abs().dropna()
    sep_mean = float(sep.mean()) if len(sep) else 0.0
    ok = (
        cov >= cfg.gates.opener_coverage_min
        and sep_mean >= cfg.gates.opener_separation_min
    )
    return GateResult(
        "GATE_OPENER_COVERAGE", ok,
        f"coverage {cov:.1%} (need {cfg.gates.opener_coverage_min:.0%}), "
        f"mean |open-close| {sep_mean:.2f} (need {cfg.gates.opener_separation_min})",
        f"median {sep.median():.2f}, p90 {sep.quantile(0.90):.2f}" if len(sep) else "",
    )


def gate_no_lookahead(feats: pd.DataFrame, walkforward: pd.DataFrame,
                      sample: int = 200) -> GateResult:
    """Every rating used for a game must come from a week whose cutoff precedes kickoff."""
    wf = walkforward[["season", "week", "as_of"]].drop_duplicates()
    merged = feats.merge(wf, on=["season", "week"], how="left")
    rng = np.random.default_rng(20260801)
    idx = rng.choice(len(merged), size=min(sample, len(merged)), replace=False)
    rows = merged.iloc[idx]
    bad = rows[
        pd.to_datetime(rows["as_of"], utc=True)
        > pd.to_datetime(rows["kickoff"], utc=True)
    ]
    return GateResult(
        "GATE_NO_LOOKAHEAD", len(bad) == 0,
        f"{len(rows)} sampled rows, {len(bad)} using ratings dated after kickoff",
    )


def gate_garbage(share: float, cfg: Config) -> GateResult:
    ok = cfg.gates.garbage_drop_min <= share <= cfg.gates.garbage_drop_max
    return GateResult(
        "GATE_GARBAGE_FILTER", ok,
        f"{share:.1%} of plays dropped "
        f"(expect {cfg.gates.garbage_drop_min:.0%}-{cfg.gates.garbage_drop_max:.0%})",
    )


def gate_unbiased(fits: dict, cfg: Config) -> GateResult:
    worst_name, worst = None, 0.0
    for name, f in fits.items():
        if abs(f.a) > worst:
            worst, worst_name = abs(f.a), name
    return GateResult(
        "GATE_UNBIASED", worst < cfg.gates.unbiased_max_abs_a,
        f"largest |a| = {worst:.3f} ({worst_name}), need < {cfg.gates.unbiased_max_abs_a}",
    )


def gate_unbiased_by_week(
    frame: pd.DataFrame, cfg: Config, min_n: int = 40
) -> GateResult:
    """|a| < 0.5 in EVERY week bucket, not merely in aggregate.

    GATE_UNBIASED is computed full-sample, which is precisely why a bias running from
    +0.62 at week 4 to +1.74 by week 13 sat underneath a passing +0.219: averaging a drift
    hides it. A constant offset larger than the selection threshold makes every game clear
    on the same side, so the sheet measures the offset rather than any game-specific view.

    Weeks thinner than `min_n` are reported but not failed on -- a two-game week cannot
    establish a bias.
    """
    rows = []
    for kind, col, mkt in (
        ("spread", "model_spread", "spread_open"),
        ("total", "model_total", "total_open"),
    ):
        d = frame.dropna(subset=[col, mkt])
        for week, grp in d.groupby("week"):
            rows.append({
                "market": kind, "week": int(week), "n": len(grp),
                "a": float((grp[col] - grp[mkt]).mean()),
            })
    tab = pd.DataFrame(rows)
    testable = tab[tab["n"] >= min_n]
    if testable.empty:
        return GateResult("GATE_UNBIASED_BY_WEEK", False, "no week bucket large enough")

    worst = testable.loc[testable["a"].abs().idxmax()]
    thin = tab[tab["n"] < min_n]
    detail = "  ".join(
        f"{r.market[:2]}wk{r.week}:{r.a:+.2f}"
        for r in testable.sort_values("a", key=abs, ascending=False).head(6).itertuples()
    )
    if len(thin):
        detail += f"   (not tested, n<{min_n}: " + ",".join(
            f"wk{int(r.week)}={r.n}" for r in thin.drop_duplicates("week").itertuples()
        ) + ")"
    return GateResult(
        "GATE_UNBIASED_BY_WEEK",
        bool(testable["a"].abs().max() < cfg.gates.unbiased_max_abs_a),
        f"worst |a| = {abs(worst['a']):.3f} ({worst['market']} wk{int(worst['week'])}, "
        f"n={int(worst['n'])}), need < {cfg.gates.unbiased_max_abs_a}",
        detail,
    )


def gate_scale(fits: dict, cfg: Config) -> GateResult:
    worst_name, worst = None, 0.0
    for name, f in fits.items():
        d = abs(f.sd_ratio - 1.0)
        if d > worst:
            worst, worst_name = d, name
    return GateResult(
        "GATE_SCALE", worst < cfg.gates.scale_ratio_tolerance,
        f"worst SD-ratio deviation {worst:.3f} ({worst_name}), "
        f"need < {cfg.gates.scale_ratio_tolerance}",
    )


def gate_blend_informative(fits: dict, cfg: Config) -> GateResult:
    best_name, best = None, 0.0
    for name, f in fits.items():
        if f.b > 0.0 and f.t_b > best:
            best, best_name = f.t_b, name
    return GateResult(
        "GATE_BLEND_INFORMATIVE", best > cfg.gates.blend_t_threshold,
        f"best positive t(b) = {best:.2f} ({best_name}), "
        f"need > {cfg.gates.blend_t_threshold}",
        "  ".join(f"{n}: b={f.b:+.4f} t={f.t_b:+.2f}" for n, f in fits.items()),
    )


def rmse_gate(
    frame: pd.DataFrame, cfg: Config, kind: str = "total", grade: str = "close"
) -> GateResult:
    """Absolute accuracy, as a HARD PRECONDITION on promotion (Phase 0D).

    The model's RMSE against the realised outcome must not exceed the market line's. This
    is deliberately stricter than the theory requires, and the reason is specific to this
    build's failure history rather than to forecasting in general.

    **What it costs.** A model can carry genuine incremental information — a positive `b`
    in the residual blend — while being less accurate on its own than the line it bets
    against. That is the normal case for a blended forecast, and this gate rejects it. The
    strictness is the conservative reading, taken deliberately.

    **Why it is worth that.** `b` has now been manufactured twice in this build without any
    information behind it: once on NFL totals, where a +1.43-point level bias presented as
    `b = +0.060` and de-biasing flipped it to -0.050, and once on NCAA totals, where
    anchoring the measurement on the opener produced `b = +0.195, t = +2.23` from a model
    that reads `b = +0.076, t = +0.89` against the close. Both artifacts inflate `b`.
    **Neither improves RMSE**, because neither adds information about the outcome. RMSE is
    the metric an errors-in-variables artifact cannot fake, which is exactly why it is the
    precondition rather than another significance test.

    Graded against the CLOSE by default. Grading absolute accuracy against the opener would
    reintroduce the anchor whose contamination motivated this gate (Appendix A).

    The market line is its own control here: `ratio = model_rmse / market_rmse`, and a
    ratio at or above 1.0 says the model adds nothing an uninformed bettor could not get
    by taking the number off the screen.
    """
    model, mkt, act = (
        ("model_total", f"total_{grade}", "actual_total") if kind == "total"
        else ("model_spread", f"spread_{grade}", "actual_margin")
    )
    d = frame.dropna(subset=[model, mkt, act])
    if len(d) < 200:
        return GateResult(
            f"GATE_RMSE_{kind.upper()}", False, f"only {len(d)} gradeable rows — cannot assess accuracy"
        )
    model_rmse = float(np.sqrt(((d[model] - d[act]) ** 2).mean()))
    mkt_rmse = float(np.sqrt(((d[mkt] - d[act]) ** 2).mean()))
    ratio = model_rmse / mkt_rmse if mkt_rmse > 0 else float("inf")
    return GateResult(
        f"GATE_RMSE_{kind.upper()}", ratio <= cfg.gates.rmse_max_ratio,
        f"{kind} RMSE {model_rmse:.3f} vs market {mkt_rmse:.3f} "
        f"(ratio {ratio:.3f}, need <= {cfg.gates.rmse_max_ratio}), n={len(d)}",
        f"graded on {grade}; market line is the zero-skill control",
    )


def gate_api_budget(calls_used: int, cfg: Config) -> GateResult:
    return GateResult(
        "GATE_API_BUDGET", calls_used < cfg.gates.api_budget_max_cold,
        f"{calls_used} calls for a cold build "
        f"(budget {cfg.gates.api_budget_max_cold})",
    )


# --------------------------------------------------------------------------------------
# Simulator-backed distribution check (GATE_KEY_NUMBERS)
# --------------------------------------------------------------------------------------
# THE MEAN PATH STILL DOES NOT GO THROUGH THE SIMULATOR -- see the module docstring. This
# section only pools SIMULATED margins to check the SHAPE of the distribution the drive
# simulator produces (the key-number spikes a linear model cannot reproduce) against the
# real historical distribution. Nothing here feeds `model_spread` or `model_total`, and
# it was added 2026-08-16 to close a gate that had been declared but never produced (see
# GATES.md, "Known gap: NCAA calibration weights are not produced by the pipeline" and the
# `GATE_CALIBRATED`/`GATE_KEY_NUMBERS` rows recorded `passed: null`).

KEY_NUMBERS = (3, 7, 10, 14, 17, 21)
TAIL_THRESHOLD = 28


def pooled_margin_pmf(
    cfg: Config,
    games: pd.DataFrame,
    drives_raw: pd.DataFrame,
    walkforward: pd.DataFrame,
    linear_frame: pd.DataFrame,
    n_games: int = 400,
    n_sims: int = 4000,
) -> dict:
    """Pool simulated margins across many historical games and compare the shape to
    history. Mirrors nfl-model's `backtest.py::pooled_margin_pmf`, with one addition NFL
    already had and NCAA did not: each sampled game's raw simulated distribution is
    recentred (`SimResult.recentered`, an importance-reweight, not a shift) onto that
    game's own `model_spread` from `linear_frame` before pooling.

    Why this exists, root-caused 2026-08-17 in `analysis/blowout_tail_diagnosis.py`: the
    raw (uncentred) simulator's own predicted mean varies only ~7.6 pts SD across
    different matchups, vs ~13.1 for the market's spread and ~11.6 for this codebase's
    gain-corrected `model_spread` -- the simulator's Monte Carlo draw variance is
    approximately right, but it is compressing genuine mismatches toward the mean, which
    is exactly what produces GATE_KEY_NUMBERS's underproduced blowout tail. NFL avoids
    this by construction: `SimResult.recentered()` is called before any consumer ever
    sees a sim margin. NCAA never wired that path in (the same missing piece
    `GATE_CALIBRATED` needs `walk_forward` to grow, per GATES.md), so both this gate and
    `project_game.py`'s printed spread were reading the uncorrected number.

    This still does not touch the mean PATH in the sense that matters for betting
    decisions -- `model_spread`/`model_total` are not written here, only read from
    `linear_frame`, which is the same walk-forward frame the RMSE/blend gates already
    grade. It changes what THIS gate measures (recentred sim shape, not raw sim shape),
    which is the point: it makes the shape test agree with the mean the model actually
    reports.

    `linear_frame` must carry `game_id` and `model_spread` (e.g. `run_backtest.py`'s
    `frame`, or the persisted `backtest_frame_default.parquet`). A game missing from it,
    or with a null `model_spread`, is dropped from the pool rather than pooled uncentred
    -- pooling a mix of recentred and raw margins would silently reintroduce the same bug
    for whichever games happened to be missing.
    """
    season = int(walkforward["season"].max())
    drive_table = build_drive_table(drives_raw, games)
    start_fp = fit_start_field_position(drive_table[drive_table["season"] < season])
    drive_model = fit_drive_model(drive_table, walkforward, cfg, as_of_season=season)
    endgame = fit_endgame_table(drive_table, cfg, as_of_season=season)
    venue_hfa = estimate_venue_hfa(games, cfg, walkforward=walkforward)

    spread_by_game = (
        linear_frame[["game_id", "model_spread"]]
        .dropna(subset=["model_spread"])
        .drop_duplicates(subset=["game_id"])
        .set_index("game_id")["model_spread"]
    )

    pool = games[
        games["season"].isin(cfg.graded_seasons)
        & games["homePoints"].notna()
        & games["awayPoints"].notna()
    ]
    pool = pool.sample(min(n_games, len(pool)), random_state=cfg.simulation.seed)

    sim_cfg = replace(cfg, simulation=replace(cfg.simulation, n_sims=n_sims))
    rng = np.random.default_rng(cfg.simulation.seed)
    # `recentered()` does not shift `sim.margins` -- it reweights the same draws
    # (`sim.weights`) so their weighted mean hits the target. So the pool must accumulate
    # a WEIGHTED histogram per game, not concatenate raw margin arrays and count: that
    # would silently discard every game's recentering and reproduce the pre-fix bug.
    pooled_pmf: dict[int, float] = {}
    n_pooled = 0
    skipped_no_target = 0
    for _, g in pool.iterrows():
        if g["game_id"] not in spread_by_game.index:
            skipped_no_target += 1
            continue
        try:
            rt = ratings_at(walkforward, g["season"], g["week"])
        except KeyError:
            continue
        if g["homeTeam"] not in rt.index or g["awayTeam"] not in rt.index:
            continue
        ctx = build_context(g, cfg, venue_hfa=venue_hfa, allow_network=False)
        sim = simulate_game(
            g["homeTeam"], g["awayTeam"], rt, drive_model, sim_cfg, start_fp,
            context_adj=ctx, endgame=endgame, rng=rng,
            neutral_site=bool(g.get("neutralSite", False)),
        )
        sim = sim.recentered(float(spread_by_game.loc[g["game_id"]]))
        vals, inv = np.unique(sim.margins, return_inverse=True)
        game_mass = np.zeros(len(vals))
        np.add.at(game_mass, inv, sim.weights)
        for v, w in zip(vals, game_mass):
            pooled_pmf[int(v)] = pooled_pmf.get(int(v), 0.0) + float(w)
        n_pooled += 1

    if skipped_no_target:
        print(f"  pooled_margin_pmf: skipped {skipped_no_target} sampled games with no "
              f"model_spread in linear_frame")
    if not n_pooled:
        return {}
    # Each game contributes equal total mass (its own weights already sum to 1), matching
    # the pre-fix behaviour where every game supplied the same n_sims equally-weighted
    # draws -- games were, and still are, weighted equally in the pool.
    return {v: w / n_pooled for v, w in pooled_pmf.items()}


def gate_key_numbers(
    sim_pmf: dict, market: pd.DataFrame, tol: float = 2.0
) -> GateResult:
    """Simulated |margin| PMF vs historical, at the college key numbers plus the tail.

    College's key-number set is not the NFL's: no missed-extra-point/safety combination
    produces the NFL's spike at 6, and college football's extra-point/two-point rates
    differ measurably from the NFL's (NEXT_SESSION 2026-08-07: colleges convert extra
    points MORE often, 0.973 vs 0.940, and go for two LESS often, 0.070 vs 0.095) --
    consequences for which margins carry mass. 17 and 21 (two- and three-score games with
    the extra point) carry real mass in college the NFL check does not look for.
    """
    done = market[market["actual_margin"].notna() & market["fbs_only"]]
    real = done["actual_margin"].abs().value_counts(normalize=True)
    real_tail_pct = 100 * float((done["actual_margin"].abs() > TAIL_THRESHOLD).mean())

    rows, worst, worst_k = [], 0.0, None
    for k in KEY_NUMBERS:
        sim_pct = 100 * (sim_pmf.get(k, 0.0) + sim_pmf.get(-k, 0.0))
        real_pct = 100 * float(real.get(k, 0.0))
        diff = abs(sim_pct - real_pct)
        rows.append(
            f"|{k}|: sim {sim_pct:5.2f}%  real {real_pct:5.2f}%  ({sim_pct - real_pct:+.2f}pp)"
        )
        if diff > worst:
            worst, worst_k = diff, str(k)

    sim_tail_pct = 100 * sum(v for m, v in sim_pmf.items() if abs(m) > TAIL_THRESHOLD)
    tail_diff = abs(sim_tail_pct - real_tail_pct)
    rows.append(
        f">{TAIL_THRESHOLD}: sim {sim_tail_pct:5.2f}%  real {real_tail_pct:5.2f}%  "
        f"({sim_tail_pct - real_tail_pct:+.2f}pp)"
    )
    if tail_diff > worst:
        worst, worst_k = tail_diff, f">{TAIL_THRESHOLD}"

    return GateResult(
        "GATE_KEY_NUMBERS",
        worst <= tol,
        f"worst gap {worst:.2f}pp at |margin|={worst_k} (tolerance {tol:.0f}pp)",
        "\n      ".join(rows),
    )


def bootstrap(results:np.ndarray,seasons=None,n_boot:int=10000,seed:int=20260801):
    if len(results) == 0:
        return float("nan"), float("nan"), float("nan")
    if seasons is not None:
        result=robust_inference.season_block_bootstrap(np.asarray(results,dtype=float),seasons,n_boot=n_boot,seed=seed,confidence=.90)
        return result.point,result.low,result.high
    rng = np.random.default_rng(seed)
    rates = rng.choice(results, size=(n_boot, len(results)), replace=True).mean(axis=1)
    return (
        float(results.mean()),
        float(np.percentile(rates, 5)),
        float(np.percentile(rates, 95)),
    )


def filtered_record(frame: pd.DataFrame, cfg: Config, kind: str, grade: str = "open"):
    """Win/loss array for positions clearing the edge threshold, pushes dropped."""
    if kind == "spread":
        model, mkt, act = "model_spread", f"spread_{grade}", "actual_margin"
        thresh = cfg.betting.min_edge_points_spread
    else:
        model, mkt, act = "model_total", f"total_{grade}", "actual_total"
        thresh = cfg.betting.min_edge_points_total

    d = frame.dropna(subset=[model, mkt, act])
    edge = (d[model] - d[mkt]).to_numpy(float)
    take_high = edge > 0
    a, k = d[act].to_numpy(float), d[mkt].to_numpy(float)
    sel = (np.abs(edge) >= thresh) & (a != k)
    if not sel.any():
        return np.array([]), d.iloc[[]]
    won = np.where(take_high[sel], a[sel] > k[sel], a[sel] < k[sel]).astype(float)
    picked = d[sel].copy()
    picked["side"] = np.where(take_high[sel], 1.0, -1.0)
    return won, picked


# --------------------------------------------------------------------------------------
# Zero-skill control -- permanent harness fixture
# --------------------------------------------------------------------------------------

def zero_skill_projection(
    frame: pd.DataFrame, anchor: str = "total_close", seed: int = 20260804,
    noise_sd: "float | None" = None,
) -> pd.DataFrame:
    """A projection with, by construction, no information: anchor + gaussian noise.

    Run through the same pick generator as the model. Its CLV is the control reading, and
    it exists because CLV can be manufactured without skill: if a projection is anchored on
    the OPENER, the opener's own noise appears in both the regressor and the response, and
    the close -- being a better estimate of the truth -- moves toward the projection's
    disagreement on average. That produces a high CLV and a positive `b` from a model that
    knows nothing.

    If this control's CLV is not near 0.50, the CLV metric is measuring line noise rather
    than skill on that anchor and must not be trusted there.
    """
    rng = np.random.default_rng(seed)
    out = frame.copy()
    base = out[anchor].astype(float)
    if noise_sd is None:
        # Match the real model's disagreement dispersion, so the control is comparable.
        noise_sd = float((out["model_total"] - out["total_open"]).std())
    out["model_total"] = base + rng.normal(0.0, noise_sd, len(out))
    return out
