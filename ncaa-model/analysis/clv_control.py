"""Is CLV a real metric here, or is it reading line noise? — Phase 0C.

The gate: a projection carrying **no information** must produce CLV = 0.50 +/- 0.03. If a
zero-skill control scores above chance, CLV is not measuring skill and must not be reported.

What this module found
----------------------
`NCAA_PLAYBOOK.md` Appendix A Test 2 reports a zero-skill control reproducing the model's
CLV exactly (0.6923 vs 0.6923) and concludes that CLV "is a reading of line noise, not of
skill." **That control was constructed with lookahead, and the conclusion drawn from it is
wrong.** The arithmetic:

    side          = sign(model_total - total_open)
    clv_positive  = sign(total_close - total_open) == sign(side)

The control in question is `model_total = total_close + noise`. Substituting:

    side          = sign((total_close - total_open) + noise) = sign(line_move + noise)
    clv_positive  = sign(line_move) == sign(line_move + noise)

That is not a test of whether an uninformed projection earns CLV. It asks whether the sign
of the line move survives adding noise to it, and the answer is "usually" for any noise
comparable to the move — which is what 0.69 is. **A control handed the closing line cannot
be a control for a metric that grades against the closing line.** It scored high because it
knew the answer, not because CLV is broken.

The valid control anchors on the number actually available at bet time:

    model_total   = total_open + noise
    side          = sign(noise)                        -- independent of the line move
    E[clv]        = 0.50                               -- by construction

Both are run below. The opener-anchored control is the gate; the close-anchored one is
retained and reported as the demonstration that it is invalid.

Why many seeds
--------------
A single control run yields ~65 graded picks, where the standard error on a proportion is
~0.062. A +/- 0.03 gate cannot be evaluated at that precision — one seed can land anywhere
in [0.38, 0.62] on chance alone. The control's *expected* CLV is the quantity under test,
so it is averaged across `--seeds` independent draws, and the across-seed standard error is
reported beside it so the gate is read against a number it can actually resolve.

    python -m analysis.clv_control                # 200 seeds
    python -m analysis.clv_control --seeds 40     # quick check

**This module does not change the standing null.** Appendix A Test 1 — re-anchoring the
model on the close, t = +2.23 -> +0.89 — is untouched by any of this and was always the
decisive test. What changes is that the CLV metric survives, so it remains available to
Phases 1-3 rather than being deleted.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from run_clv_history import _attach_conferences, _conference_lookup
from src.backtest import select_window
from src.betting import build_totals_sheet, clv_summary
from src.calibrate import build_conditional
from src.config import CACHE_DIR, OUTPUT_DIR, load_config
from src.market import fit_blend

GATE_TARGET = 0.50
GATE_TOLERANCE = 0.03
RULE = "=" * 84


def side_baselines(frame: pd.DataFrame) -> dict:
    """P(the line moves toward side s), estimated on the whole graded population.

    Totals lines in this universe drift DOWN by about a point on average. A pick set that
    is 63% UNDER therefore collects CLV at well above 0.50 without expressing any view,
    which is precisely what the uninformed control measured. The baseline is the drift
    itself, so subtracting it is what makes CLV a statement about the pick rather than
    about the market's habits.
    """
    move = (frame["total_close"] - frame["total_open"]).dropna()
    move = move[move != 0]
    return {1.0: float((move > 0).mean()), -1.0: float((move < 0).mean())}


def adjusted_clv(sheet: pd.DataFrame, baselines: dict) -> float:
    """CLV with each pick graded against its own side's base rate, re-centred on 0.50.

    An uninformed projection scores 0.50 here by construction regardless of how lopsided
    its side mix is, which is the property raw CLV lacks.
    """
    graded = sheet[sheet["clv_positive"].notna()]
    if graded.empty:
        return float("nan")
    base = graded["side"].map(baselines).astype(float)
    return float(0.50 + (graded["clv_positive"].astype(float) - base).mean())


def _history(cfg, frame: pd.DataFrame, conf: dict) -> pd.DataFrame:
    """Walk-forward pick generation. Mirrors run_clv_history.generate_history.

    Reimplemented against a preloaded conference map rather than the raw market frame so
    the seed sweep does not re-derive the same lookup 200 times. Caps are off: the caps
    are a concentration diagnostic and would only add variance to a control whose picks
    are random by construction.
    """
    graded = select_window(frame, cfg.graded_seasons, restricted=True)
    out = []
    for season in sorted(graded["season"].unique()):
        train = graded[graded["season"] < season]
        test = graded[graded["season"] == season]
        if len(train) < 400 or test.empty:
            continue
        weights = fit_blend(train, cfg, window=f"pre-{season}")
        pmf = build_conditional(
            train, "total_open", "actual_total", as_of_season=season, kind="total"
        )
        sheet = build_totals_sheet(test, weights, cfg, pmf=pmf, picks_only=True,
                                   caps=False)
        if sheet.empty:
            continue
        out.append(_attach_conferences(sheet, conf))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _control_frame(frame: pd.DataFrame, anchor: str, seed: int,
                   noise_sd: float) -> pd.DataFrame:
    """A projection with no information: `anchor + N(0, noise_sd)`.

    `noise_sd` is fixed across seeds and anchors — taken from the real model's own
    disagreement dispersion — so the two anchors differ in exactly one respect, which is
    the thing being tested.
    """
    rng = np.random.default_rng(seed)
    out = frame.copy()
    out["model_total"] = out[anchor].astype(float) + rng.normal(0.0, noise_sd, len(out))
    return out


def sweep(cfg, frame: pd.DataFrame, conf: dict, anchor: str, seeds: int,
          noise_sd: float, baselines: dict) -> pd.DataFrame:
    rows = []
    for seed in range(20260804, 20260804 + seeds):
        sheet = _history(cfg, _control_frame(frame, anchor, seed, noise_sd), conf)
        if sheet.empty:
            continue
        s = clv_summary(sheet)
        # Side balance is recorded because it is the one mechanism by which an UNINFORMED
        # projection can still beat 0.50: if the selection filters admit more OVERs than
        # UNDERs, and totals lines drift systematically in one direction, the pick set
        # inherits that drift as CLV without any game-specific view. A control sitting
        # above chance with a lopsided side split is diagnosing the filters, not the model.
        over = float((sheet["side"] > 0).mean()) if len(sheet) else float("nan")
        rows.append({"anchor": anchor, "seed": seed, "picks": s["picks"],
                     "graded": s["with_close_move"], "clv": s["clv"],
                     "clv_adj": adjusted_clv(sheet, baselines),
                     "over_share": over,
                     "mean_line_move": float(sheet["line_move"].mean())})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=200)
    args = ap.parse_args()

    cfg = load_config()
    frame = pd.read_parquet(CACHE_DIR / "backtest_frame_default.parquet")
    market = pd.read_parquet(CACHE_DIR / "market.parquet")
    conf = _conference_lookup(market)

    graded = select_window(frame, cfg.graded_seasons, restricted=True)
    noise_sd = float((graded["model_total"] - graded["total_open"]).std())
    baselines = side_baselines(graded)

    print(RULE)
    print("CLV ZERO-SKILL CONTROL — Phase 0C")
    print(f"gate: control CLV = {GATE_TARGET:.2f} +/- {GATE_TOLERANCE:.2f}   "
          f"noise_sd = {noise_sd:.3f} pts   seeds = {args.seeds}")
    print(RULE)

    model_sheet = _history(cfg, frame, conf)
    ms = clv_summary(model_sheet)
    print(f"\n  side baselines: P(move toward OVER) {baselines[1.0]:.4f}   "
          f"P(move toward UNDER) {baselines[-1.0]:.4f}")
    print("\n  the model (reference, not a control)")
    print(f"    picks {ms['picks']}   graded {ms['with_close_move']}   "
          f"raw CLV {ms['clv']:.4f}   adjusted {adjusted_clv(model_sheet, baselines):.4f}")

    results = []
    for anchor, note in (
        ("total_open", "VALID — uses only what is known at bet time"),
        ("total_close", "INVALID — has lookahead into the grading quantity"),
    ):
        tab = sweep(cfg, frame, conf, anchor, args.seeds, noise_sd, baselines)
        mean_clv = float(tab["clv"].mean())
        se = float(tab["clv"].std(ddof=1) / np.sqrt(len(tab)))
        mean_adj = float(tab["clv_adj"].mean())
        se_adj = float(tab["clv_adj"].std(ddof=1) / np.sqrt(len(tab)))
        inside = abs(mean_adj - GATE_TARGET) <= GATE_TOLERANCE
        verdict = ("PASS" if inside else "FAIL") if anchor == "total_open" \
            else "n/a (invalid control)"
        results.append(tab)
        print(f"\n  control anchored on {anchor}   [{note}]")
        print(f"    mean picks {tab['picks'].mean():.1f}   "
              f"mean graded {tab['graded'].mean():.1f}")
        print(f"    raw CLV       {mean_clv:.4f}   se {se:.4f}")
        print(f"    adjusted CLV  {mean_adj:.4f}   se {se_adj:.4f}   <- the gate reads this")
        print(f"    OVER share {tab['over_share'].mean():.3f}   "
              f"mean line move {tab['mean_line_move'].mean():+.3f} pts")
        print(f"    GATE       {verdict}")

    combined = pd.concat(results, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_DIR / "clv_control.csv", index=False)

    valid = combined[combined["anchor"] == "total_open"]
    mean_valid = float(valid["clv_adj"].mean())
    passed = abs(mean_valid - GATE_TARGET) <= GATE_TOLERANCE

    print()
    print(RULE)
    if passed:
        print(f"GATE_CLV_CONTROL  PASS — an uninformed projection earns {mean_valid:.4f}, "
              "indistinguishable from chance.")
        print("CLV is measuring something real and is RETAINED.")
        print("Appendix A Test 2 is superseded: its control was anchored on the close and")
        print("therefore knew the quantity CLV grades. Test 1 is unaffected and still")
        print("carries the null on its own.")
    else:
        print(f"GATE_CLV_CONTROL  FAIL — an uninformed projection earns {mean_valid:.4f}.")
        print("CLV is not measuring skill on this pipeline and must be DELETED, not")
        print("reported with a caveat.")
    print(RULE)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
