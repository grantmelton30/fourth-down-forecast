"""Excel export, shared by both repos. Sport is a config argument.

    python export.py --sport nfl --season 2025 --week 10
    python export.py --sport ncaa --season 2026 --week 5

THE BETS SHEET IS GATED. `adapter.bets_allowed()` reads the blend artifact and is the only
thing that can turn picks on. When it is False the workbook still builds -- ratings, games,
diagnostics, all of it -- and a `No Bets` sheet explains why in plain language. There is no
flag to override it, because the point of the gate is that it is not subject to the
operator's mood on a given Saturday.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .sport import SportAdapter, load_adapter
except ImportError:  # run as a script rather than imported as a package
    from sport import SportAdapter, load_adapter

# xlsxwriter number formats, defined once so every sheet agrees.
FMT = {
    "spread": "+0.0;-0.0",
    "total": "0.0",
    "rating": "0.0000",
    "pct": "0.0%",
}


def write_workbook(
    adapter: SportAdapter,
    season: int,
    week: int,
    out_path: "Path | None" = None,
) -> Path:
    """Build the workbook for one slate. Returns the path written."""
    label = adapter.profile.label.lower()
    out_path = Path(out_path) if out_path else (
        adapter.profile.repo / "output" / f"{label}_week_{season}_{week}.xlsx"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ratings = adapter.ratings(season, week)
    games = adapter.games(season, week)
    gates = adapter.gate_table()
    blend = adapter.blend_weights()
    backtest = adapter.backtest_frame()

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as xl:
        styles = _styles(xl.book)
        _sheet_ratings(xl, styles, ratings, adapter)
        _sheet_games(xl, styles, games, adapter)
        _sheet_diagnostics(xl, styles, gates, blend, backtest, adapter)
        if adapter.bets_allowed():
            # Reached only when a component has cleared GATE_BLEND_INFORMATIVE.
            _sheet_bets_placeholder(xl, styles)
        else:
            _sheet_no_bets(xl, styles, adapter)

    return out_path


# --------------------------------------------------------------------------------------
# Sheets
# --------------------------------------------------------------------------------------

def _styles(book) -> dict:
    return {
        "title": book.add_format({"bold": True, "font_size": 14}),
        "header": book.add_format({
            "bold": True, "bg_color": "#1F3864", "font_color": "white",
            "border": 1, "text_wrap": True, "valign": "vcenter",
        }),
        "body": book.add_format({"border": 1}),
        "rating": book.add_format({"border": 1, "num_format": FMT["rating"]}),
        "spread": book.add_format({"border": 1, "num_format": FMT["spread"]}),
        "total": book.add_format({"border": 1, "num_format": FMT["total"]}),
        "pct": book.add_format({"border": 1, "num_format": FMT["pct"]}),
        "pass": book.add_format({
            "border": 1, "bg_color": "#C6EFCE", "font_color": "#006100", "bold": True,
        }),
        "fail": book.add_format({
            "border": 1, "bg_color": "#FFC7CE", "font_color": "#9C0006", "bold": True,
        }),
        "warn": book.add_format({
            "bg_color": "#FFF2CC", "text_wrap": True, "valign": "top", "border": 1,
        }),
        "note": book.add_format({
            "italic": True, "font_color": "#555555", "text_wrap": True, "valign": "top",
        }),
    }


def _write_table(ws, styles, df: pd.DataFrame, start_row: int = 0,
                 index_label: str = "") -> int:
    """Write a frame with a styled header. Returns the next free row."""
    if df is None or df.empty:
        ws.write(start_row, 0, "(no data)", styles["note"])
        return start_row + 2

    frame = df.reset_index() if (df.index.name or index_label) else df.copy()
    if index_label and len(frame.columns):
        frame = frame.rename(columns={frame.columns[0]: index_label})

    for c, name in enumerate(frame.columns):
        ws.write(start_row, c, str(name), styles["header"])
    for r, (_, row) in enumerate(frame.iterrows(), start=start_row + 1):
        for c, name in enumerate(frame.columns):
            val = row[name]
            lname = str(name).lower()
            fmt = styles["body"]
            if isinstance(val, (float, np.floating)):
                if "rating" in lname or "change" in lname:
                    fmt = styles["rating"]
                elif "prob" in lname:
                    fmt = styles["pct"]
                elif any(k in lname for k in ("spread", "edge", "margin", "result")):
                    fmt = styles["spread"]
                else:
                    fmt = styles["total"]
            if val is None or (isinstance(val, (float, np.floating)) and pd.isna(val)):
                ws.write_blank(r, c, None, fmt)
            elif isinstance(val, (int, float, np.integer, np.floating)):
                ws.write_number(r, c, float(val), fmt)
            else:
                ws.write(r, c, str(val), fmt)

    for c, name in enumerate(frame.columns):
        ws.set_column(c, c, max(12, min(26, len(str(name)) + 4)))
    ws.freeze_panes(start_row + 1, 0)
    return start_row + len(frame) + 3


def _sheet_ratings(xl, styles, ratings: pd.DataFrame, adapter: SportAdapter) -> None:
    """The sheet you look at most often to sanity-check whether the model believes
    something absurd."""
    ws = xl.book.add_worksheet("Ratings")
    xl.sheets["Ratings"] = ws
    ws.write(0, 0, f"{adapter.profile.label} team ratings", styles["title"])
    ws.write(
        1, 0,
        f"Units: {adapter.profile.rating_units}. Higher off_rating = better offense; "
        "higher def_rating = WORSE defense. net_rating already accounts for that sign, "
        "so higher is better and it is the column to sort by.",
        styles["note"],
    )
    _write_table(ws, styles, ratings, start_row=3, index_label="team")


def _sheet_games(xl, styles, games: pd.DataFrame, adapter: SportAdapter) -> None:
    """Every game on the slate. No picks here even when the gate passes -- picks get their
    own sheet so this one stays a neutral view."""
    ws = xl.book.add_worksheet("All Games")
    xl.sheets["All Games"] = ws
    ws.write(0, 0, f"{adapter.profile.label} slate", styles["title"])
    ws.write(
        1, 0,
        f"Market lines are {adapter.profile.market_line_basis}. {adapter.profile.notes}",
        styles["note"],
    )
    _write_table(ws, styles, games, start_row=3)


def _sheet_diagnostics(xl, styles, gates, blend, backtest, adapter) -> None:
    """Gates, blend coefficients and backtest evidence, so the workbook carries its own
    provenance rather than asking you to trust it."""
    ws = xl.book.add_worksheet("Diagnostics")
    xl.sheets["Diagnostics"] = ws
    ws.set_column(0, 0, 32)
    ws.set_column(1, 1, 16)
    ws.set_column(2, 2, 76)
    ws.write(0, 0, f"{adapter.profile.label} diagnostics", styles["title"])

    row = 2
    ws.write(row, 0, "Acceptance gates", styles["header"])
    row += 1
    if gates is not None and len(gates):
        for _, g in gates.iterrows():
            ws.write(row, 0, str(g["gate"]), styles["body"])
            ws.write(row, 1, str(g["status"]),
                     styles["pass"] if g["status"] == "PASS" else styles["fail"])
            ws.write(row, 2, str(g["observed"]), styles["body"])
            row += 1
    else:
        ws.write(row, 0, "(no gate artifact -- run the backtest)", styles["note"])
        row += 1

    row += 1
    ws.write(row, 0, "Blend coefficients (residual form, free intercept)", styles["header"])
    row += 1
    if blend:
        for k in ("a", "b_model", "t_model", "b_model_total",
                  "t_model_total", "r_squared", "resid_sd", "n"):
            if blend.get(k) is None:
                continue
            ws.write(row, 0, k, styles["body"])
            val = blend[k]
            if isinstance(val, (int, float, np.integer, np.floating)):
                ws.write_number(row, 1, float(val), styles["rating"])
            else:
                ws.write(row, 1, str(val), styles["body"])
            row += 1
    else:
        ws.write(row, 0, "(no blend artifact)", styles["note"])
        row += 1

    if backtest is not None and len(backtest):
        row += 1
        ws.write(row, 0, "Backtest summary", styles["header"])
        row += 1
        for k, v in _backtest_summary(backtest).items():
            ws.write(row, 0, k, styles["body"])
            ws.write_number(row, 1, float(v), styles["rating"])
            row += 1


def _backtest_summary(frame: pd.DataFrame) -> dict:
    def rmse(a, b):
        m = a.notna() & b.notna()
        return float(np.sqrt(((a[m] - b[m]) ** 2).mean())) if m.any() else float("nan")

    out = {
        "graded games": float(len(frame)),
        "model spread RMSE": rmse(frame["model_spread"], frame["actual_margin"]),
        "market spread RMSE": rmse(frame["market_spread"], frame["actual_margin"]),
        "model SD": float(frame["model_spread"].std()),
        "market SD": float(frame["market_spread"].std()),
        "SD ratio": float(frame["model_spread"].std() / frame["market_spread"].std()),
    }
    if "elo_spread" in frame.columns:
        out["elo spread RMSE"] = rmse(frame["elo_spread"], frame["actual_margin"])
    if "model_total" in frame.columns:
        out["model total RMSE"] = rmse(frame["model_total"], frame["actual_total"])
        out["market total RMSE"] = rmse(frame["market_total"], frame["actual_total"])
    return out


def _sheet_no_bets(xl, styles, adapter: SportAdapter) -> None:
    """Exists precisely so a failing gate is loud rather than silent."""
    ws = xl.book.add_worksheet("No Bets")
    xl.sheets["No Bets"] = ws
    ws.set_column(0, 0, 110)
    ws.write(0, 0, "NO BETS EMITTED", styles["title"])
    ws.write(2, 0, adapter.bet_block_reason(), styles["warn"])
    ws.write(
        4, 0,
        "This is not an error and not a missing feature. The bet sheet is gated behind "
        "GATE_BLEND_INFORMATIVE in both repos. Ratings, slate and diagnostics are fully "
        "populated -- use them to inspect the model, not to place bets.",
        styles["note"],
    )
    ws.write(
        6, 0,
        "To change this outcome you need a component whose held-out blend t-statistic "
        "clears 2.0. Re-running the same backtest after a null result cannot produce "
        "that; it can only manufacture a false positive.",
        styles["note"],
    )


def _sheet_bets_placeholder(xl, styles) -> None:
    """Only reachable once a gate passes. Selection logic lives in each repo's
    betting.py; this sheet is its presentation surface."""
    ws = xl.book.add_worksheet("Bets")
    xl.sheets["Bets"] = ws
    ws.set_column(0, 0, 110)
    ws.write(0, 0, "BETS", styles["title"])
    ws.write(
        2, 0,
        "GATE_BLEND_INFORMATIVE has passed for this sport. Wire "
        "betting.build_bet_sheet() into this sheet.",
        styles["warn"],
    )


# --------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Excel export for the NFL/NCAA models")
    ap.add_argument("--sport", default="nfl", help="nfl | ncaa")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    adapter = load_adapter(args.sport)
    if not adapter.available:
        print(f"{args.sport}: repo not built yet at {adapter.profile.repo}")
        return 1

    path = write_workbook(adapter, args.season, args.week, args.out)
    print(f"wrote {path}")
    print("bets sheet: "
          + ("INCLUDED" if adapter.bets_allowed() else "BLOCKED (gate failing)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
