"""Why does the drive simulator underproduce blowouts (GATE_KEY_NUMBERS, >28 tail)?

GATES.md recorded the standing hypothesis (from NEXT_SESSION.md, pre-dating this repo):
the shared drive-count clamp `(_MIN_DRIVES, _MAX_DRIVES) = (8, 16)` in shared/sim_core.py
truncates college's highest-possession games, and a capped possession count mechanically
caps how lopsided a simulated score can get.

This measures that hypothesis directly against the real drive table (no API calls --
everything here is already cached) instead of trusting it, per this repo's own rule.

Verdict so far (see the first block below): THE CLAMP IS NOT THE CAUSE. In the real data,
only ~1.6% of games have either team run more than 16 drives, and that rate is LOWER in
blowout games (|margin|>28) than in close ones -- 1.2% vs 1.7%. If the clamp were the
mechanism, blowouts would need MORE high-drive-count games than average; they need fewer.
Mean max-team-drives barely differs by margin bucket (12.0-12.5 across the board). The
clamp essentially never binds and does not track where margins actually come from.

Second block: measures margin dispersion under the SIMULATOR directly (re-running
pooled_margin_pmf's own loop, capturing the raw array instead of collapsing to a PMF) and
compares it to the real margin distribution's shape, not just its mean and SD, to test
the alternative hypothesis: within-game drive outcomes are i.i.d. given a fixed net_epa in
the simulator, but real blowouts likely involve positively correlated drives (a team run
that is scoring well is more likely to keep scoring well within the same game -- fatigue,
morale, a shaken defense) -- which would thin the simulator's tail without the mean or
overall SD needing to be wrong at all.

Verdict on block 2, 2026-08-17: THAT ALTERNATIVE HYPOTHESIS IS ALSO NOT IT, and the real
answer is more basic than either. On a 140-game pooled sample: skew and excess kurtosis are
close between real and simulated (0.023 vs 0.037, -0.331 vs 0.069) -- the SHAPE is not
badly wrong. What is wrong is a plain SD deficit, and decomposing pooled sim SD (16.79) into
its two sources shows exactly where: within-game (Monte Carlo draw) SD is 14.98, across-game
(spread of each game's own simulated MEAN) SD is only 7.56.

Benchmarked against the market on the full 3,472-4,068 game backtest frame
(`backtest_frame_default.parquet`): spread_close SD is 13.085, and actual_margin SD is
20.068. Backing out the game-day-randomness component implied by those two numbers --
sqrt(20.068^2 - 13.085^2) = 15.2 -- lands almost exactly on the simulator's own within-game
SD of 14.98. **The simulator's Monte Carlo draw variance is approximately correct.** The
deficit is entirely in the across-game term: the simulator's own predicted MEAN margin
varies only 7.56 pts SD across different matchups, vs. 13.085 for the market's spread and
11.630 for this codebase's OWN linear projection (`model_spread`, gain-corrected). The
simulator's rating-to-margin mapping compresses real mismatches far more than either
benchmark -- that is why extreme (>28 pt) games are underproduced: the model isn't drawing
noisier games, it's failing to recognize which games are true mismatches in the first
place.

Tested and REJECTED as the mechanism: `fit_drive_model`'s L2 regularization (`C`, default
1.0). Refit at C=1, 10, 100 (two orders of magnitude, cache bypassed) on the same 140-game
sample: across-game SD moved 7.56 -> 7.71 -> 7.69. Flat. Not a shrinkage-strength problem.

Most likely explanation, NOT YET CONFIRMED: **there is no recentring step on the NCAA
side.** `nfl-model` computes `cover_prob_home`/`over_prob` via `SimResult.recentered()`,
which shifts the simulator's margin distribution onto the better-calibrated L1
(gain-corrected linear) mean before it is used for anything user-facing -- see
NEXT_SESSION.md's L1/L2 section. `ncaa-model` has no `recenter` call anywhere in the
package (`grep -rl recenter src/` is empty). `project_game.py` prints `sim.mean_margin`
-- the simulator's own raw, compressed mean -- directly as "PROJECTED SPREAD", and
`pooled_margin_pmf` (below) pools the same raw, uncentred `sim.margins` and is exactly what
`GATE_KEY_NUMBERS` grades. Both consumers see the compressed mean. NFL structurally cannot
have this bug because it never lets the raw sim mean reach a consumer unrecentred; NCAA
never built that step, most likely because `GATE_CALIBRATED` (the gate that would have
forced simulate_game into the live walk-forward loop) was also never built for NCAA --
recorded in GATES.md as "not built... needs `drives`/`walkforward` threaded through
`walk_forward`... a real signature change". The two missing pieces are the same missing
piece. Implementing `recentered()` for NCAA would fix `project_game.py`'s displayed number
and would very likely narrow (not necessarily fully close -- the linear mean's own SD,
11.630, still trails the market's 13.085) the GATE_KEY_NUMBERS tail gap, without touching
`model_spread`/`model_total`, which already do not use the simulator's mean. Not
implemented in this pass -- it changes what `GATE_KEY_NUMBERS` measures (recentred sim
margins, not raw ones), which is a real semantic decision this file's own rules say should
not be made silently.

RECENTRING WAS IMPLEMENTED, 2026-08-17 (separate commit, `pooled_margin_pmf` now takes
`linear_frame` and recentres per game). Measured effect: >28 tail gap -8.33pp -> -3.63pp.
Real, and it does not fully close, exactly as predicted above (the linear mean's own SD
still trails the market's).

WHY IT DOES NOT FULLY CLOSE, root-caused 2026-08-17: swept the fitted drive model's own
`net_epa -> E[points/drive]` response across the real observed rating range
([-0.14, +0.30], from actual `off_rating`/`def_rating` extremes) and read off the slope.
It is not linear -- `MultinomialModel.probabilities()` is a softmax over
`[net_epa, fp, fp^2, home, net_epa*fp]` by construction, and softmax saturates. Marginal
E[points/drive] per unit net_epa peaks at 13.4 near net_epa=+0.06 (a middling favorite)
and falls to 6.8 at net_epa=-0.14 (worst offense vs. best defense) and 9.0 at
net_epa=+0.24 (best offense vs. worst defense) -- roughly half the model's peak
sensitivity spent exactly where real blowout mismatches live. This is a property of the
multinomial-softmax parameterization itself, present at any regularization strength,
which is exactly why sweeping `C` (above) moved nothing: `C` shrinks coefficients
uniformly, it does not change the functional form's saturation. Confirms the recentring
fix could only partially close the gap -- the underlying drive model structurally
compresses the exact games it needs to spread out the most. Candidate fix, not attempted:
add `net_epa**2` (or a spline) to the design matrix so the model has room to counteract
its own saturation at the extremes -- requires refitting `fit_drive_model` and
revalidating every gate that reads from it, out of scope for this pass.

TESTED, 2026-08-17: THE CANDIDATE FIX DOES NOT WORK, and the reason is worth having
gotten wrong first. Refit the real, cached training data (196,371 drives) with
`net_epa**2` added to the design matrix and re-swept the same net_epa range. The
marginal-sensitivity ratio (min slope / max slope, the saturation measurement from PART 4)
barely moved: 0.40 baseline vs 0.41 with the squared term. `net_epa**2` is symmetric in
net_epa, so its coefficient mostly rescales the curve's overall width around net_epa=0,
not the SHAPE that causes saturation at both extremes -- and the deeper reason no
polynomial feature could fix this is that softmax outputs are bounded in [0, 1] by
construction, for any input, for any set of features. A team cannot have a >100% chance
of scoring a touchdown no matter what the design matrix contains; saturation here is not
a flexibility problem this model failed to solve, it is a mathematical property of
predicting a bounded probability at all. Ruling this out redirects any future effort away
from "give the drive model more features" -- correctly, per this measurement -- and
toward either accepting this as a structural limit of drive-based simulation, or a
genuinely different model family for the mean path, neither attempted here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import drives as D
from src import ingest
from src.backtest import pooled_margin_pmf, KEY_NUMBERS, TAIL_THRESHOLD
from src.cfbd_client import BudgetedCFBD
from src.config import CACHE_DIR, load_config
from src.venues import attach_venues, load_venues


def clamp_hypothesis_check() -> pd.DataFrame:
    games = pd.read_parquet(CACHE_DIR / "games_2019_2025.parquet")
    frames = []
    for year in range(2019, 2026):
        d = pd.json_normalize(json.loads((CACHE_DIR / f"drives_{year}.json").read_text()))
        d["season"] = year
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True).rename(columns={"gameId": "game_id"})
    dt = D.build_drive_table(raw, games[["game_id", "week"]])

    g = games[
        games["homePoints"].notna() & games["awayPoints"].notna()
        & games["season"].between(2021, 2025)
    ].copy()
    g["margin"] = (g["homePoints"] - g["awayPoints"]).abs()

    counts = dt.groupby(["game_id", "offense"]).size().rename("n_drives").reset_index()
    per_game_max = counts.groupby("game_id")["n_drives"].max().rename("max_team_drives")
    g = g.merge(per_game_max, on="game_id", how="inner")

    print(f"games with drive data, 2021-2025: {len(g):,}")
    print(f"P(either team's drive count > 16): {(g['max_team_drives'] > 16).mean():.4f}")
    blowout = g["margin"] > TAIL_THRESHOLD
    print(f"blowout rate (|margin|>{TAIL_THRESHOLD}): {blowout.mean():.4f}")
    print(f"P(max_team_drives>16 | blowout):     {g.loc[blowout, 'max_team_drives'].gt(16).mean():.4f}")
    print(f"P(max_team_drives>16 | not blowout): {g.loc[~blowout, 'max_team_drives'].gt(16).mean():.4f}")
    bins = [0, 3, 7, 14, 21, 28, 999]
    g["margin_bucket"] = pd.cut(g["margin"], bins)
    tbl = g.groupby("margin_bucket", observed=True)["max_team_drives"].agg(
        mean="mean", median="median", p_over_16=lambda x: (x > 16).mean(), n="size"
    )
    print(tbl)
    print()
    if g.loc[blowout, "max_team_drives"].gt(16).mean() <= g.loc[~blowout, "max_team_drives"].gt(16).mean():
        print("VERDICT: clamp hypothesis REJECTED -- blowouts do not run more drives than "
              "close games; the clamp virtually never binds (~1.6% of all games) and binds "
              "LESS often in blowouts, not more.")
    return g


def real_margin_moments(g: pd.DataFrame) -> None:
    m = g["homePoints"] - g["awayPoints"] if "homePoints" in g else None


def distribution_shape_check(cfg, games_with_venues, drives_raw, wf, n_games=600, n_sims=4000) -> None:
    """Re-run the pooled simulation, but keep the raw margins instead of collapsing to a
    PMF, and compare shape (not just mean/SD) against the real distribution over the same
    sampled games."""
    from src.context import build_context, estimate_venue_hfa
    from src.drive_model import fit_drive_model, fit_endgame_table
    from src.drives import build_drive_table, fit_start_field_position
    from src.ratings import ratings_at
    from src.simulate import simulate_game
    from dataclasses import replace

    season = int(wf["season"].max())
    drive_table = build_drive_table(drives_raw, games_with_venues)
    start_fp = fit_start_field_position(drive_table[drive_table["season"] < season])
    drive_model = fit_drive_model(drive_table, wf, cfg, as_of_season=season)
    endgame = fit_endgame_table(drive_table, cfg, as_of_season=season)
    venue_hfa = estimate_venue_hfa(games_with_venues, cfg, walkforward=wf)

    pool = games_with_venues[
        games_with_venues["season"].isin(cfg.graded_seasons)
        & games_with_venues["homePoints"].notna()
        & games_with_venues["awayPoints"].notna()
    ]
    pool = pool.sample(min(n_games, len(pool)), random_state=cfg.simulation.seed)

    sim_cfg = replace(cfg, simulation=replace(cfg.simulation, n_sims=n_sims))
    rng = np.random.default_rng(cfg.simulation.seed)

    sim_margins_per_game = []  # one MEAN per sampled game (matches "one game" granularity)
    real_margins = []
    for _, g in pool.iterrows():
        try:
            rt = ratings_at(wf, g["season"], g["week"])
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
        sim_margins_per_game.append(sim.margins)  # full n_sims draws for this one game
        real_margins.append(float(g["homePoints"]) - float(g["awayPoints"]))

    pooled_sim = np.concatenate(sim_margins_per_game)
    real = np.array(real_margins)

    print(f"games simulated: {len(real_margins)}")
    print()
    print(f"{'':20s} {'real':>12s} {'sim (pooled)':>14s}")
    print(f"{'mean |margin|':20s} {np.abs(real).mean():12.2f} {np.abs(pooled_sim).mean():14.2f}")
    print(f"{'SD':20s} {real.std():12.2f} {pooled_sim.std():14.2f}")
    print(f"{'skew':20s} {stats.skew(real):12.3f} {stats.skew(pooled_sim):14.3f}")
    print(f"{'excess kurtosis':20s} {stats.kurtosis(real):12.3f} {stats.kurtosis(pooled_sim):14.3f}")
    print(f"{'P(|margin|>28)':20s} {(np.abs(real) > 28).mean():12.4f} {(np.abs(pooled_sim) > 28).mean():14.4f}")
    print()

    # Per-game mean margin, vs the SD of a single game's own simulated distribution --
    # tests whether the simulator's GAME-LEVEL spread (how differently two teams are
    # projected) is realistic, separate from its WITHIN-GAME spread (Monte Carlo draw
    # variance for one matchup).
    per_game_sd = np.array([float(np.std(m)) for m in sim_margins_per_game])
    per_game_mean = np.array([float(np.mean(m)) for m in sim_margins_per_game])
    print(f"mean per-game simulated SD (within-game draw spread): {per_game_sd.mean():.2f}")
    print(f"SD of per-game simulated MEANS (across-game spread):  {per_game_mean.std():.2f}")
    print(f"real |margin| SD (across games):                     {real.std():.2f}")

    bf_path = CACHE_DIR / "backtest_frame_default.parquet"
    if bf_path.exists():
        bf = pd.read_parquet(bf_path)
        market_sd = bf["spread_close"].std()
        actual_sd = bf["actual_margin"].std()
        linear_sd = bf["model_spread"].std()
        implied_within_game_sd = np.sqrt(actual_sd**2 - market_sd**2)
        print()
        print(f"market spread_close SD (full sample, n={bf['spread_close'].notna().sum()}): "
              f"{market_sd:.3f}")
        print(f"actual_margin SD (full sample, n={bf['actual_margin'].notna().sum()}):      "
              f"{actual_sd:.3f}")
        print(f"model_spread SD, gain-corrected linear (full sample):     {linear_sd:.3f}")
        print(f"implied real within-game SD, sqrt(actual^2-market^2):     "
              f"{implied_within_game_sd:.3f}  (vs. sim within-game SD {per_game_sd.mean():.2f} above)")
        print(f"-> sim's within-game draw variance looks about right; the deficit is almost"
              f" entirely in across-game dispersion ({per_game_mean.std():.2f} vs market's "
              f"{market_sd:.3f} and the linear model's own {linear_sd:.3f}).")


def regularization_sweep(cfg, games_with_venues, drives_raw, wf, n_games=140, n_sims=2000,
                          Cs=(1.0, 10.0, 100.0)) -> None:
    """Tests whether fit_drive_model's L2 strength C explains the across-game SD deficit.
    Verdict, 2026-08-17: no -- SD is flat across two orders of magnitude of C."""
    from src.drive_model import fit_drive_model, fit_endgame_table
    from src.drives import build_drive_table, fit_start_field_position
    from src.ratings import ratings_at
    from src.context import build_context, estimate_venue_hfa
    from src.simulate import simulate_game
    from dataclasses import replace

    season = int(wf["season"].max())
    drive_table = build_drive_table(drives_raw, games_with_venues)
    start_fp = fit_start_field_position(drive_table[drive_table["season"] < season])
    endgame = fit_endgame_table(drive_table, cfg, as_of_season=season)
    venue_hfa = estimate_venue_hfa(games_with_venues, cfg, walkforward=wf)

    pool = games_with_venues[
        games_with_venues["season"].isin(cfg.graded_seasons)
        & games_with_venues["homePoints"].notna()
        & games_with_venues["awayPoints"].notna()
    ]
    pool = pool.sample(min(n_games, len(pool)), random_state=cfg.simulation.seed)

    for C in Cs:
        drive_model = fit_drive_model(drive_table, wf, cfg, as_of_season=season, C=C, cache=False)
        sim_cfg = replace(cfg, simulation=replace(cfg.simulation, n_sims=n_sims))
        rng = np.random.default_rng(cfg.simulation.seed)
        means = []
        for _, g in pool.iterrows():
            try:
                rt = ratings_at(wf, g["season"], g["week"])
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
            means.append(float(np.mean(sim.margins)))
        means = np.array(means)
        print(f"C={C:6.1f}  n={len(means):4d}  SD of per-game means={means.std():.2f}")


def saturation_sweep(cfg) -> None:
    """Is the drive model's net_epa -> E[points/drive] response linear, or does the
    softmax parameterization saturate at the extremes where real mismatches live?
    Verdict, 2026-08-17: it saturates -- marginal sensitivity roughly halves at either
    end of the real observed rating range versus its peak near a middling favorite."""
    from src.config import CACHE_DIR
    from sim_core import MultinomialModel

    model = MultinomialModel.from_json((CACHE_DIR / "drive_model_2019_2026.json").read_text())
    net_epa_range = np.linspace(-0.20, 0.30, 26)
    fp = np.full_like(net_epa_range, model.fp_mean)
    probs = model.probabilities(net_epa_range, fp, 0.0)
    idx = {c: i for i, c in enumerate(model.classes)}
    exp_pts = probs[:, idx["TD"]] * 6.94 + probs[:, idx["FG"]] * 3.0
    slope = np.gradient(exp_pts, net_epa_range)
    print(f"{'net_epa':>8s} {'E[pts/drive]':>13s} {'marginal slope':>15s}")
    for ne, ep, s in zip(net_epa_range, exp_pts, slope):
        print(f"{ne:8.3f} {ep:13.3f} {s:15.3f}")
    peak = net_epa_range[np.argmax(slope)]
    print(f"\npeak marginal sensitivity {slope.max():.2f} at net_epa={peak:+.3f}; "
          f"falls to {slope[0]:.2f} at the low end ({net_epa_range[0]:+.3f}) and "
          f"{slope[-1]:.2f} at the high end ({net_epa_range[-1]:+.3f}).")


def net_epa_squared_experiment(cfg, games_with_venues, drives_raw, wf) -> None:
    """Does adding net_epa**2 to the design matrix let the drive model counteract its own
    softmax saturation? Verdict, 2026-08-17: no. Refits the real training data directly
    (sklearn, bypassing the cached MultinomialModel) with and without the squared term and
    compares the same marginal-sensitivity ratio PART 4 uses. The reason it fails is more
    useful than the negative result itself: net_epa**2 is symmetric, so its coefficient
    mostly rescales the curve's width around net_epa=0, not the saturating SHAPE -- and no
    polynomial feature can fix that shape, because softmax outputs are bounded in [0, 1]
    by construction regardless of what the design matrix contains. This rules out "add
    more features" as a path forward for this specific gap, correctly, rather than by
    assumption."""
    from sklearn.linear_model import LogisticRegression
    from sim_core import DRIVE_CLASSES
    from src.drive_model import build_training_features
    from src.drives import build_drive_table

    drive_table = build_drive_table(drives_raw, games_with_venues)
    season = int(wf["season"].max())
    train = drive_table[(drive_table["season"] >= cfg.seasons.train_start)
                        & (drive_table["season"] < season)]
    feat = build_training_features(train, wf, cfg)
    print(f"training rows: {len(feat):,}")

    net = feat["net_epa"].to_numpy(float)
    fp_mean, fp_scale = 50.0, 25.0
    fp = (feat["start_yardline_100"].to_numpy(float) - fp_mean) / fp_scale
    home = feat["is_home_offense"].to_numpy(float)
    y = feat["result"].to_numpy()

    def fit(with_square: bool):
        cols = [net, fp, fp ** 2, home, net * fp]
        if with_square:
            cols.append(net ** 2)
        return LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000).fit(
            np.column_stack(cols), y)

    base, squared = fit(False), fit(True)
    order = [list(base.classes_).index(c) for c in DRIVE_CLASSES]
    tdi, fgi = list(DRIVE_CLASSES).index("TD"), list(DRIVE_CLASSES).index("FG")

    def exp_pts(clf, net_range, with_square):
        zeros = np.zeros_like(net_range)
        cols = [net_range, zeros, zeros, zeros, net_range * zeros]
        if with_square:
            cols.append(net_range ** 2)
        logits = np.column_stack(cols) @ clf.coef_[order].T + clf.intercept_[order]
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        return p[:, tdi] * 6.94 + p[:, fgi] * 3.0

    net_range = np.linspace(-0.20, 0.30, 26)
    slope_base = np.gradient(exp_pts(base, net_range, False), net_range)
    slope_sq = np.gradient(exp_pts(squared, net_range, True), net_range)
    ratio_base = slope_base.min() / slope_base.max()
    ratio_sq = slope_sq.min() / slope_sq.max()
    print(f"baseline slope range:      {slope_base.min():.2f} to {slope_base.max():.2f}"
          f"  (min/max ratio {ratio_base:.2f})")
    print(f"+net_epa**2 slope range:   {slope_sq.min():.2f} to {slope_sq.max():.2f}"
          f"  (min/max ratio {ratio_sq:.2f})")
    print(f"\nratio moved {ratio_base:.2f} -> {ratio_sq:.2f}: "
          f"{'meaningful change' if abs(ratio_sq - ratio_base) > 0.05 else 'no meaningful change'}")


def main() -> int:
    print("=" * 90)
    print("PART 1 -- the recorded clamp hypothesis, measured directly")
    print("=" * 90)
    clamp_hypothesis_check()

    print()
    print("=" * 90)
    print("PART 2 -- distribution shape: simulator vs real")
    print("=" * 90)
    cfg = load_config()
    client = BudgetedCFBD(cfg)
    games = ingest.load_games(client, cfg.all_seasons)
    drives_raw = ingest.load_drives(client, cfg.all_seasons)
    game_off = ingest.build_game_offense(
        ingest.load_plays(client, games, cfg.all_seasons), drives_raw, games, cfg
    )
    from src.ratings import build_walkforward
    wf = build_walkforward(game_off, games, cfg)
    games_with_venues = attach_venues(games, load_venues(client))
    print(f"CFBD calls used: {client.calls_used}")
    distribution_shape_check(cfg, games_with_venues, drives_raw, wf)

    print()
    print("=" * 90)
    print("PART 3 -- is drive-model L2 regularization (C) the cause of the deficit?")
    print("=" * 90)
    regularization_sweep(cfg, games_with_venues, drives_raw, wf)

    print()
    print("=" * 90)
    print("PART 4 -- does the drive model's net_epa response saturate at the extremes?")
    print("=" * 90)
    saturation_sweep(cfg)

    print()
    print("=" * 90)
    print("PART 5 -- does adding net_epa**2 let the model counteract its own saturation?")
    print("=" * 90)
    net_epa_squared_experiment(cfg, games_with_venues, drives_raw, wf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
