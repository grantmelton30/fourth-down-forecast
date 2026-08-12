"""Excel export for the NCAA totals sheet.

Four sheets: Bets (capped picks), All Games, Ratings, Diagnostics. The Bets sheet exists
only because GATE_BLEND_INFORMATIVE passed on totals; spreads are absent by design, and
there is no stake column anywhere in the workbook.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR, Config
from .market import BlendWeights


def write_workbook(
    picks: pd.DataFrame,
    weights: BlendWeights,
    cfg: Config,
    season: int,
    weeks: list,
    frame: pd.DataFrame,
) -> Path:
    path = OUTPUT_DIR / f"ncaa_totals_{season}_wk{weeks[0]}-{weeks[-1]}.xlsx"
    with pd.ExcelWriter(path, engine="xlsxwriter") as xl:
        book = xl.book
        title = book.add_format({"bold": True, "font_size": 14})
        hdr = book.add_format({"bold": True, "bg_color": "#1F3864", "font_color": "white",
                               "border": 1, "text_wrap": True})
        body = book.add_format({"border": 1})
        num = book.add_format({"border": 1, "num_format": "0.00"})
        pct = book.add_format({"border": 1, "num_format": "0.0%"})
        note = book.add_format({"italic": True, "font_color": "#555555",
                                "text_wrap": True, "valign": "top"})
        warn = book.add_format({"bg_color": "#FFF2CC", "text_wrap": True,
                                "valign": "top", "border": 1})

        _table(xl, "Bets", picks, hdr, body, num, pct, title, note,
               "TOTALS ONLY. Spreads are dark: b is indistinguishable from zero on them. "
               "No stake column -- stakes go live only on positive CLV.")

        allg = frame[frame["season"] == season]
        keep = [c for c in ("week", "away_team", "home_team", "total_open", "total_close",
                            "model_total", "actual_total", "restricted", "cross_tier")
                if c in allg.columns]
        _table(xl, "All Games", allg[keep], hdr, body, num, pct, title, note,
               f"Every graded {season} game, including those not bet.")

        ws = book.add_worksheet("Diagnostics")
        xl.sheets["Diagnostics"] = ws
        ws.set_column(0, 0, 42)
        ws.set_column(1, 1, 58)
        ws.write(0, 0, "NCAA totals — diagnostics", title)
        rows = [
            ("b_model_total", f"{weights.b_model_total:+.4f}"),
            ("t(b_model_total)", f"{weights.t_model_total:+.2f}"),
            ("a (intercept, totals)", f"{weights.a_total:+.3f}"),
            ("b_model_spread", f"{weights.b_model_spread:+.4f} (dark)"),
            ("t(b_model_spread)", f"{weights.t_model_spread:+.2f} (dark)"),
            ("graded vs", "OPENING line; close retained for CLV"),
            # CLV figures deliberately NOT baked in here. They were hardcoded as
            # literals, went stale the moment the close-anchored null landed, and
            # rendered known-false numbers into every workbook. CLV is reported by
            # run_clv_history.py alongside its zero-skill control, or not at all.
            ("excluded weeks (totals)", str(cfg.betting.exclude_weeks_total)),
            ("max team pick share", f"{cfg.betting.max_team_pick_share:.0%}"),
            ("max conference pick share", f"{cfg.betting.max_conference_pick_share:.0%}"),
            ("min edge (blended pts)", f"{cfg.betting.min_edge_points_total}"),
            ("min cover prob", f"{cfg.betting.min_cover_prob}"),
        ]
        for i, (k, v) in enumerate(rows, start=2):
            ws.write(i, 0, k, body)
            ws.write(i, 1, v, body)
        ws.write(len(rows) + 3, 0,
                 "STANDING RULE: stakes go live only on positive CLV. This workbook sizes "
                 "nothing.", warn)
    return path


def _table(xl, name, df, hdr, body, num, pct, title, note, caption):
    ws = xl.book.add_worksheet(name)
    xl.sheets[name] = ws
    ws.write(0, 0, name, title)
    ws.write(1, 0, caption, note)
    if df is None or df.empty:
        ws.write(3, 0, "(no rows)", note)
        return
    for c, col in enumerate(df.columns):
        ws.write(3, c, str(col), hdr)
        ws.set_column(c, c, max(11, min(24, len(str(col)) + 4)))
    for r, (_, row) in enumerate(df.iterrows(), start=4):
        for c, col in enumerate(df.columns):
            v = row[col]
            fmt = pct if "prob" in str(col).lower() else (
                num if isinstance(v, (float, np.floating)) else body)
            if v is None or (isinstance(v, (float, np.floating)) and pd.isna(v)):
                ws.write_blank(r, c, None, fmt)
            elif isinstance(v, (int, float, np.integer, np.floating)):
                ws.write_number(r, c, float(v), fmt)
            else:
                ws.write(r, c, str(v), fmt)
    ws.freeze_panes(4, 0)
