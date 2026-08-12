"""Streamlit viewer, shared by both repos. Sport is a config argument.

    streamlit run view.py -- --sport nfl
    streamlit run view.py -- --sport ncaa

Three tabs: Ratings, Game explorer, Diagnostics.

Picks are never rendered here unless `adapter.bets_allowed()` is true, and that is decided
in sport.py from the stored blend artifact -- not by anything in this file. The NFL model
returns False, so this viewer shows its numbers and refuses to show selections.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sport import load_adapter  # noqa: E402

try:
    import altair as alt
    HAVE_ALT = True
except ImportError:  # streamlit ships altair, but do not hard-fail if it is absent
    HAVE_ALT = False


# --------------------------------------------------------------------------------------
# Config argument
# --------------------------------------------------------------------------------------

def resolve_sport() -> str:
    """`--sport` after streamlit's `--`, else $SPORT_MODEL, else nfl."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", default=None)
    known, _ = ap.parse_known_args()
    if known.sport:
        return known.sport
    return os.environ.get("SPORT_MODEL", "nfl")


@st.cache_resource(show_spinner="Loading model...")
def get_adapter(sport: str):
    return load_adapter(sport)


# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------

def tab_ratings(adapter, season: int, week: int) -> None:
    st.subheader(f"Team ratings — {season} week {week}")
    ratings = adapter.ratings(season, week)
    if ratings is None or ratings.empty:
        st.info("No ratings for this period yet.")
        return

    st.markdown(
        '<div class="subtle">Higher <b>net rating</b> is better. Ratings are in '
        f'{adapter.profile.rating_units}.</div>',
        unsafe_allow_html=True,
    )
    with st.expander("How to read these columns"):
        st.markdown(
            "- **off_rating** — higher is a better offense.\n"
            "- **def_rating** — higher is a *worse* defense. It is the coefficient on the "
            "opposing defense in a model of offensive EPA, so the sign runs the other way.\n"
            "- **net_rating** — already accounts for that sign, so higher is better."
        )

    num = [c for c in ratings.select_dtypes("number").columns
           if "rank" not in c and "games" not in c]
    # A native bar column rather than Styler.background_gradient, which would drag in
    # matplotlib for one colour ramp.
    col_cfg = {}
    if "net_rating" in ratings:
        span = float(ratings["net_rating"].abs().max() or 1.0)
        col_cfg["net_rating"] = st.column_config.ProgressColumn(
            "net_rating", format="%.4f", min_value=-span, max_value=span,
        )
    query = st.text_input(
        "Filter teams", "", placeholder="type part of a team name",
        key=f"ratings_filter_{season}_{week}",
    )
    shown = ratings
    if query:
        shown = ratings[ratings.index.astype(str).str.contains(query, case=False)]
        if shown.empty:
            st.info(f"No team matches {query!r}.")
            shown = ratings

    st.dataframe(
        shown.style.format({c: "{:+.4f}" for c in num}),
        width='stretch', height=420, column_config=col_cfg,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Teams rated", len(ratings))
    if "net_rating" in ratings:
        c2.metric("Best net rating", str(ratings["net_rating"].idxmax()),
                  f"{ratings['net_rating'].max():+.3f}")
        c3.metric("Worst net rating", str(ratings["net_rating"].idxmin()),
                  f"{ratings['net_rating'].min():+.3f}")

    if HAVE_ALT and "net_rating" in ratings:
        chart_df = ratings.reset_index()
        chart_df = chart_df.rename(columns={chart_df.columns[0]: "team"})
        st.altair_chart(
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("net_rating:Q", title="net rating"),
                y=alt.Y("team:N", sort="-x", title=None),
                color=alt.condition(
                    alt.datum.net_rating > 0,
                    alt.value("#2e7d32"), alt.value("#c62828"),
                ),
                tooltip=[c for c in chart_df.columns],
            )
            .properties(height=max(320, 18 * len(chart_df))),
            width='stretch',
        )

    st.download_button(
        "Download ratings CSV",
        ratings.to_csv().encode(),
        file_name=f"{adapter.profile.key}_ratings_{season}_wk{week}.csv",
        mime="text/csv",
    )


def tab_game_explorer(adapter, season: int, week: int) -> None:
    st.subheader(f"Game explorer — {season} week {week}")
    games = adapter.games(season, week)
    if games is None or games.empty:
        st.info(
            "No games available for this period. The Game explorer needs the schedule "
            "and the simulator; for a repo still at the ratings stage this tab stays "
            "dark by design."
        )
        return

    labels = {}
    for r in games.itertuples(index=False):
        sp = getattr(r, "spread_line", None)
        suffix = f"  ({sp:+.1f})" if sp is not None and pd.notna(sp) else ""
        labels[f"{r.away_team} @ {r.home_team}{suffix}"] = r.game_id

    choice = st.selectbox("Game", list(labels))
    game_id = labels[choice]
    row = games[games["game_id"] == game_id].iloc[0]

    with st.spinner("Simulating..."):
        sim = adapter.simulate(game_id)
    if sim is None:
        st.warning("The simulator is not available for this game.")
        return

    # Calibrated simulations retain the original score lattice and attach probability
    # weights.  Every display must therefore use the weighted SimResult summaries.
    model_spread = float(sim.mean_margin)
    model_total = float(sim.mean_total)
    market_spread = row.get("spread_line")
    market_total = row.get("total_line")

    # The scoreline, from the simulated means. mean_total = home + away and mean_margin =
    # home - away, so the two halves are recovered by the same identity the simulator uses.
    home_pts = (model_total + model_spread) / 2.0
    away_pts = (model_total - model_spread) / 2.0
    n_sims = len(np.asarray(sim.margins))
    _matchup_card(row["away_team"], row["home_team"], away_pts, home_pts,
                  f"total {model_total:.1f} · {n_sims:,} simulations")

    p_home = float(sim.weights[np.asarray(sim.margins) > 0].sum())
    _win_prob_bar(p_home, row["home_team"], row["away_team"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model spread", f"{model_spread:+.2f}",
              _fmt(row.get("spread_line"), "market {:+.1f}"), delta_color="off")
    m2.metric("Model total", f"{model_total:.2f}",
              _fmt(row.get("total_line"), "market {:.1f}"), delta_color="off")
    if market_spread is not None and pd.notna(market_spread):
        m3.metric("Spread disagreement", f"{model_spread - float(market_spread):+.2f} pts")
    if market_total is not None and pd.notna(market_total):
        m4.metric("Total disagreement", f"{model_total - float(market_total):+.2f} pts")

    if pd.notna(row.get("result")) or pd.notna(row.get("total")):
        f1, f2 = st.columns(2)
        f1.metric("Final margin", _fmt(row.get("result"), "{:+.0f}"))
        f2.metric("Final total", _fmt(row.get("total"), "{:.0f}"))

    st.markdown(
        '<div class="subtle">Positive spread = home favoured, matching the market '
        'convention. The model line is the simulated mean — not a pick, and not blended '
        'with the market here.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        _histogram(sim.margins, "Simulated margin (home - away)",
                   {"market": market_spread, "model": model_spread},
                   key_numbers=adapter.profile.key_numbers, weights=sim.weights)
    with right:
        _histogram(sim.totals, "Simulated total",
                   {"market": market_total, "model": model_total}, weights=sim.weights)
    st.caption(
        "Amber bars mark the key numbers — the margins football actually lands on. Those "
        "spikes are what simulating drive by drive buys you over drawing from a bell curve."
    )

    if market_spread is not None and pd.notna(market_spread):
        st.markdown(
            "**Cover probabilities** (conditional on the bet resolving — pushes excluded)"
        )
        line = float(market_spread)
        probs = pd.DataFrame({
            "side": [f"{row['home_team']} (home)", f"{row['away_team']} (away)"],
            "cover_prob": [sim.cover_prob(line, "home"), sim.cover_prob(line, "away")],
        })
        if market_total is not None and pd.notna(market_total):
            t = float(market_total)
            probs = pd.concat([probs, pd.DataFrame({
                "side": [f"Over {t:.1f}", f"Under {t:.1f}"],
                "cover_prob": [sim.total_prob(t, "over"), sim.total_prob(t, "under")],
            })], ignore_index=True)
        _prob_bars(probs)

    if not adapter.bets_allowed():
        # Kept, deliberately: the page must never let a cover probability read as a
        # recommendation. Now a quiet line rather than a full-width info panel repeating
        # the gate text that is already one click away in the status bar.
        st.markdown(
            '<div class="subtle">Distribution readouts, not recommendations — this model '
            'does not publish selections.</div>',
            unsafe_allow_html=True,
        )

    margin_pmf = sim.margin_pmf()
    st.markdown("**Key numbers in this simulation**")
    # Plain field names: Vega-Lite parses '.', '[' and ']' as accessors, so a column
    # literally called "|margin|" is a needless bet on how it handles punctuation.
    kn_df = pd.DataFrame({
        "margin": list(adapter.profile.key_numbers),
        "pct": [
            100 * (float(margin_pmf.get(k, 0.0)) + float(margin_pmf.get(-k, 0.0)))
            for k in adapter.profile.key_numbers
        ],
    })
    if HAVE_ALT:
        st.altair_chart(
            alt.Chart(kn_df).mark_bar(cornerRadiusEnd=3, color=KEY_COLOR).encode(
                # Ordinal axes rotate labels by default; these are one or two digits and
                # read better flat.
                x=alt.X("margin:O", title="|margin|",
                        axis=alt.Axis(labelAngle=0)),
                y=alt.Y("pct:Q", title="% of simulations"),
                tooltip=[alt.Tooltip("margin:O", title="|margin|"),
                         alt.Tooltip("pct:Q", title="% of sims", format=".2f")],
            ).properties(height=190),
            width='stretch',
        )
    else:
        st.dataframe(
            kn_df.rename(columns={"margin": "|margin|", "pct": "simulated %"})
                 .style.format({"simulated %": "{:.2f}"}),
            width='stretch', hide_index=True,
        )


def _season_weeks(adapter, season: int) -> list:
    p = adapter.rating_periods()
    if p is None or p.empty:
        return []
    return sorted(int(w) for w in p.loc[p["season"] == season, "week"].unique())


def tab_team(adapter, season: int, week: int) -> None:
    """One team, whole season: rating trajectory and every opponent.

    Presentation only. Everything here is the same `ratings()` and `games()` the other tabs
    read, sliced by team instead of by week -- no new model quantity is computed.
    """
    ratings = adapter.ratings(season, week)
    if ratings is None or ratings.empty:
        st.info("No ratings for this period yet.")
        return

    teams = sorted(str(t) for t in ratings.index)
    team = st.selectbox(f"Team — {season}", teams, key=f"team_pick_{season}")

    row = ratings.loc[team]
    cols = st.columns(4)
    for col, name in zip(cols, ("net_rating", "off_rating", "def_rating", "pace_rating")):
        if name in ratings.columns:
            # def_rating is inverted: higher means a WORSE defense, so rank ascending.
            rank = int(ratings[name].rank(ascending=(name == "def_rating")).loc[team])
            col.metric(name.replace("_", " "), f"{float(row[name]):+.4f}",
                       f"#{rank} of {len(ratings)}", delta_color="off")

    # --- rating trajectory ------------------------------------------------------------
    weeks = _season_weeks(adapter, season)
    traj = []
    for w in weeks:
        r = adapter.ratings(season, w)
        if r is not None and not r.empty and team in r.index and "net_rating" in r.columns:
            traj.append({"week": w, "net_rating": float(r.loc[team, "net_rating"])})
    if HAVE_ALT and len(traj) > 1:
        tdf = pd.DataFrame(traj)
        st.markdown("**Rating through the season**")
        line = alt.Chart(tdf).mark_line(point=True, strokeWidth=2.5,
                                        color=BAR_COLOR).encode(
            x=alt.X("week:Q", title="week", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("net_rating:Q", title="net rating"),
            tooltip=[alt.Tooltip("week:Q"), alt.Tooltip("net_rating:Q", format="+.4f")],
        )
        zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(
            strokeDash=[4, 4], color="grey").encode(y="y:Q")
        st.altair_chart((zero + line).properties(height=220), width='stretch')

    # --- the schedule -------------------------------------------------------------------
    rows = []
    for w in weeks:
        g = adapter.games(season, w)
        if g is None or g.empty or "home_team" not in g.columns:
            continue
        mine = g[(g["home_team"] == team) | (g["away_team"] == team)]
        for r in mine.itertuples(index=False):
            home = getattr(r, "home_team", None) == team
            opp = getattr(r, "away_team" if home else "home_team", "—")
            spread = getattr(r, "spread_line", None)
            margin = getattr(r, "result", None)
            # Both are quoted home-perspective; flip for an away game so every row reads
            # from THIS team's point of view.
            if spread is not None and pd.notna(spread):
                spread = float(spread) if home else -float(spread)
            if margin is not None and pd.notna(margin):
                margin = float(margin) if home else -float(margin)
            kick = getattr(r, "kickoff", None)
            rows.append({
                "week": w,
                "date": (pd.to_datetime(kick).strftime("%b %d")
                         if kick is not None and pd.notna(kick) else "—"),
                "site": "vs" if home else "@",
                "opponent": opp,
                "line": spread,
                "total": getattr(r, "total_line", None),
                "margin": margin,
                "result": ("—" if margin is None or pd.isna(margin)
                           else "W" if margin > 0 else "L" if margin < 0 else "T"),
            })

    st.markdown(f"**{team} — {season} schedule**")
    if not rows:
        st.info("No scheduled games found for this team in this season.")
        return
    sched = pd.DataFrame(rows)
    # Coerce before styling: unplayed games carry Python None, which lands in an object
    # column, and Styler's na_rep only replaces NaN/NA -- so the table printed a literal
    # "None" in every future row.
    for c in ("line", "total", "margin"):
        sched[c] = pd.to_numeric(sched[c], errors="coerce")
    played = sched[sched["result"] != "—"]
    if len(played):
        c1, c2, c3 = st.columns(3)
        c1.metric("Record", f"{int((played['result'] == 'W').sum())}-"
                            f"{int((played['result'] == 'L').sum())}")
        c2.metric("Games played", f"{len(played)} of {len(sched)}")
        c3.metric("Avg margin", f"{played['margin'].mean():+.1f}")
    # Formatted to strings here rather than via Styler.format(na_rep=...): Streamlit does
    # not honour na_rep through st.dataframe, so every unplayed game printed a literal
    # "None". Schedules are already in week order, so losing numeric sort costs nothing.
    show = sched.copy()
    for c, spec in (("line", "{:+.1f}"), ("total", "{:.1f}"), ("margin", "{:+.0f}")):
        show[c] = sched[c].map(lambda v, s=spec: s.format(v) if pd.notna(v) else "—")
    st.dataframe(show, width='stretch', hide_index=True,
                 height=min(560, 45 + 35 * len(show)))
    st.caption(
        "`line` and `margin` are from this team's perspective — a negative line means this "
        "team is favoured. Market numbers and results only; no projection is run here."
    )
    st.download_button(
        "Download schedule CSV", sched.to_csv(index=False).encode(),
        file_name=f"{adapter.profile.key}_{team.replace(' ', '_')}_{season}.csv",
        mime="text/csv",
    )


def tab_diagnostics(adapter) -> None:
    st.subheader("Diagnostics")

    gates = adapter.gate_table()
    if gates is not None and len(gates):
        st.markdown("**Acceptance gates**")
        st.dataframe(
            gates.style.map(
                lambda v: (
                    "background-color:#c6efce;color:#006100;font-weight:bold"
                    if v == "PASS" else
                    "background-color:#ffc7ce;color:#9c0006;font-weight:bold"
                    if v == "FAIL" else ""
                ),
                subset=["status"],
            ),
            width='stretch', hide_index=True,
        )
    else:
        st.info("No gate artifact — run the backtest.")

    blend = adapter.blend_weights()
    if blend:
        st.markdown("**Blend coefficients** (residual form, free intercept)")
        st.caption(
            "`b` is the shrinkage factor on the model's disagreement with the market: "
            "b = 0.30 means when the model disagrees by 3 points, ~0.9 of that is real. "
            "`t` decides whether any of it is real at all."
        )
        st.dataframe(
            pd.DataFrame([
                {"coefficient": k, "value": float(v)}
                for k, v in blend.items()
                if isinstance(v, (int, float)) and k != "n"
            ]).style.format({"value": "{:+.4f}"}),
            width='stretch', hide_index=True,
        )

    frame = adapter.backtest_frame()
    if frame is None or not len(frame):
        st.info("No backtest frame yet.")
        return

    st.markdown("**Backtest**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Graded games", f"{len(frame):,}")
    mr = _rmse(frame["model_spread"], frame["actual_margin"])
    kr = _rmse(frame["market_spread"], frame["actual_margin"])
    c2.metric("Model spread RMSE", f"{mr:.3f}")
    c3.metric("Market spread RMSE", f"{kr:.3f}", f"{mr - kr:+.3f} vs model")
    sd_ratio = float(frame["model_spread"].std() / frame["market_spread"].std())
    c4.metric("SD ratio", f"{sd_ratio:.3f}", "target 1.05-1.10")

    if sd_ratio > 1.0:
        noise = float(np.sqrt(max(
            frame["model_spread"].std() ** 2 - frame["market_spread"].std() ** 2, 0.0
        )))
        st.caption(
            f"Implied projection noise: **{noise:.2f} points**. A projection cannot "
            "resolve disagreements smaller than its own error."
        )

    if HAVE_ALT:
        st.markdown("**Model vs market, per game**")
        samp = frame.sample(min(1200, len(frame)), random_state=0)
        scatter = alt.Chart(samp).mark_circle(opacity=0.35).encode(
            x=alt.X("market_spread:Q", title="market spread"),
            y=alt.Y("model_spread:Q", title="model spread"),
            tooltip=[c for c in ("season", "week", "home", "away", "market_spread",
                                 "model_spread") if c in samp.columns],
        ).properties(height=340)
        diag = alt.Chart(pd.DataFrame({"x": [-20.0, 20.0]})).mark_line(
            strokeDash=[4, 4], color="grey"
        ).encode(x="x:Q", y="x:Q")
        st.altair_chart(scatter + diag, width='stretch')

    st.markdown("**Calibration** — predicted cover probability vs realized rate")
    calib = _calibration(frame)
    if len(calib):
        st.dataframe(
            calib.style.format({
                "predicted": "{:.3f}", "realized": "{:.3f}", "abs_error": "{:.3f}",
            }),
            width='stretch', hide_index=True,
        )
        st.caption(
            "Systematic drift between predicted and realized here is the most common "
            "cause of a backtest that looks good and an account that bleeds."
        )

    st.markdown("**Per-season**")
    rows = []
    for season, g in frame.groupby("season"):
        rows.append({
            "season": int(season),
            "games": len(g),
            "model RMSE": _rmse(g["model_spread"], g["actual_margin"]),
            "market RMSE": _rmse(g["market_spread"], g["actual_margin"]),
        })
    st.dataframe(
        pd.DataFrame(rows).style.format({
            "model RMSE": "{:.3f}", "market RMSE": "{:.3f}",
        }),
        width='stretch', hide_index=True,
    )


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _fmt(v, spec: str) -> str:
    return spec.format(float(v)) if v is not None and pd.notna(v) else "—"


def _rmse(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    return float(np.sqrt(((a[m] - b[m]) ** 2).mean())) if m.any() else float("nan")


def _calibration(frame: pd.DataFrame, width: float = 0.05) -> pd.DataFrame:
    if "cover_prob_home" not in frame.columns:
        return pd.DataFrame()
    d = frame[frame["actual_margin"] != frame["market_spread"]].copy()
    d["outcome"] = (d["actual_margin"] > d["market_spread"]).astype(float)
    d["bin"] = pd.cut(d["cover_prob_home"], np.arange(0, 1 + width, width),
                      include_lowest=True)
    out = d.groupby("bin", observed=True).agg(
        n=("outcome", "size"),
        predicted=("cover_prob_home", "mean"),
        realized=("outcome", "mean"),
    ).reset_index()
    out["abs_error"] = (out["realized"] - out["predicted"]).abs()
    out["bin"] = out["bin"].astype(str)
    return out[out["n"] >= 20]


MODEL_COLOR = "#0f9f91"
MARKET_COLOR = "#e8663f"
BAR_COLOR = "#3f7fba"
KEY_COLOR = "#e6a23c"


def _inject_css() -> None:
    """Presentation only.

    Everything here is colour, spacing and weight. Deliberately no solid page background
    and no fixed text colour: Streamlit ships a light and a dark theme, and hardcoding
    either makes the other unreadable. Fills and borders are rgba over whatever the theme
    provides, so both stay legible.
    """
    st.markdown(
        """
        <style>
        :root {
            --field: #0f9f91; --sky: #3f7fba; --amber: #e6a23c;
            --ink-soft: rgba(128,128,128,.68); --line: rgba(128,128,128,.20);
        }
        .stApp { background-image: linear-gradient(180deg, rgba(63,127,186,.045), transparent 18rem); }
        .product-header { padding: .3rem 0 1rem; border-bottom: 1px solid var(--line); margin-bottom: .75rem; }
        .product-kicker { color: var(--field); font-size: .69rem; font-weight: 800; letter-spacing: .18em; }
        .product-title { font-size: clamp(1.75rem, 3vw, 2.65rem); font-weight: 820; letter-spacing: -.045em; line-height: 1.05; }
        .product-subtitle { margin-top: .35rem; color: var(--ink-soft); font-size: .91rem; }
        div[data-testid="stMetric"] {
            background: rgba(128, 128, 128, 0.055);
            border: 1px solid var(--line); border-top: 2px solid rgba(15,159,145,.56);
            border-radius: 8px; padding: 11px 14px 10px;
        }
        div[data-testid="stMetricLabel"] p {
            font-size: 0.74rem; font-weight: 600;
            letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.72;
        }
        div[data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 700; }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
        button[data-baseweb="tab"] { font-weight: 650; letter-spacing: .01em; padding: .7rem 1rem; }
        button[data-baseweb="tab"][aria-selected="true"] { color: var(--field); }
        div[data-testid="stSegmentedControl"] { max-width: 310px; }
        section[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
        section[data-testid="stSidebar"] > div { background: rgba(128,128,128,.035); }
        div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
        .stDownloadButton button, .stButton button { border-radius: 7px; font-weight: 650; }
        h1 { font-weight: 800; letter-spacing: -0.02em; }
        h3 { font-weight: 700; letter-spacing: -0.01em; }
        .matchup {
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 10px;
            padding: 16px 20px;
            margin-bottom: 14px;
            background: linear-gradient(110deg, rgba(15,159,145,.10), rgba(63,127,186,.06));
            box-shadow: inset 3px 0 0 rgba(15,159,145,.72);
        }
        .matchup .teams { font-size: 1.32rem; font-weight: 700; letter-spacing: -0.01em; }
        .matchup .score { font-size: 2.05rem; font-weight: 800; letter-spacing: -0.02em; }
        .matchup .sub   { font-size: 0.82rem; opacity: 0.72; margin-top: 2px; }
        .statusbar {
            display: flex; align-items: center; gap: 9px;
            border: 1px solid rgba(128,128,128,0.22);
            border-radius: 8px; padding: 9px 14px; margin: 2px 0 14px 0;
            background: rgba(128,128,128,0.06); font-size: 0.90rem;
        }
        .statusbar .dot { width: 9px; height: 9px; border-radius: 50%; flex: 0 0 auto; }
        .statusbar .muted { opacity: 0.62; }
        .subtle { opacity: 0.62; font-size: 0.80rem; line-height: 1.45; }
        div.block-container { padding-top: 1.7rem; max-width: 1480px; }
        @media (max-width: 700px) {
            .product-title { font-size: 1.85rem; }
            .product-subtitle { max-width: 28rem; }
            div.block-container { padding-top: 1rem; padding-left: 1rem; padding-right: 1rem; }
            button[data-baseweb="tab"] { padding: .6rem .55rem; font-size: .82rem; }
            .statusbar { align-items: flex-start; flex-wrap: wrap; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _status_bar(adapter, validate_gates: bool = True) -> None:
    """One-line model status, with the technical reason one click away.

    This replaced a full-width warning banner quoting gate names at the reader before they
    had seen a single number. The SUBSTANCE is unchanged and must stay that way -- picks are
    still gated on `adapter.bets_allowed()` and the page still says plainly when it will not
    show them. What changed is that the gate identifiers now live inside the expander and in
    Diagnostics, where someone looking for them will find them, rather than in the first
    thing a reader sees.
    """
    # Ratings and team pages cannot publish a selection. Keep their first render fast and
    # fail closed; full artifact validation runs when a market-facing page is selected.
    ok = adapter.bets_allowed() if validate_gates else False
    label = "Picks enabled" if ok else "Projections only"
    color = MODEL_COLOR if ok else "#c8952b"
    st.markdown(
        f"""<div class="statusbar">
          <span class="dot" style="background:{color}"></span>
          <strong>{label}</strong>
          <span class="muted">· {adapter.profile.label} · graded against the
          {adapter.profile.market_line_basis} line</span>
        </div>""",
        unsafe_allow_html=True,
    )
    if not ok:
        with st.expander("Why this model does not publish picks"):
            st.markdown(
                "This model is **less accurate than the betting market** it would be "
                "betting into, so it publishes projections and declines to publish "
                "selections. A model that only bets when it has an edge, and finds none, "
                "is behaving correctly.\n\n"
                "Full gate detail is in the **Diagnostics** tab."
            )
            st.caption(
                adapter.bet_block_reason()
                if validate_gates
                else "Promotion-gate validation is deferred until Game explorer or "
                     "Diagnostics is opened. This page remains projections-only."
            )


def _matchup_card(away: str, home: str, away_pts: float, home_pts: float,
                  subtitle: str) -> None:
    """The projected scoreline as a scoreboard rather than four undifferentiated tiles.

    Renders the simulated means it is handed. Computes nothing.
    """
    fav = home if home_pts >= away_pts else away
    edge = abs(home_pts - away_pts)
    st.markdown(
        f"""
        <div class="matchup">
          <div class="teams">{away} <span style="opacity:.5">at</span> {home}</div>
          <div class="score">{away_pts:.1f} <span style="opacity:.4">–</span> {home_pts:.1f}</div>
          <div class="sub">projected · {fav} by {edge:.1f} · {subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _win_prob_bar(p_home: float, home: str, away: str) -> None:
    """One stacked bar for win probability. `p_home` is passed in, not derived here."""
    if not HAVE_ALT:
        st.write(f"{home} win probability: {p_home:.1%}")
        return
    # A bare two-colour bar does not say which colour is which team, and the reader has no
    # way to guess. Label it in text rather than relying on the fill.
    st.markdown(
        f"""<div style="font-size:0.82rem;margin-bottom:2px">
        <span style="color:{MODEL_COLOR};font-weight:700">■</span>
        <strong>{home}</strong> (home) {p_home:.1%}
        <span style="opacity:.45"> · </span>
        <span style="color:#8c8c8c;font-weight:700">■</span>
        <strong>{away}</strong> (away) {1.0 - p_home:.1%}
        <span style="opacity:.6"> — win probability</span>
        </div>""",
        unsafe_allow_html=True,
    )
    df = pd.DataFrame({
        "team": [home, away],
        "prob": [p_home, 1.0 - p_home],
        "side": ["home", "away"],
    })
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadius=4, height=30)
        .encode(
            x=alt.X("prob:Q", stack="normalize", title=None,
                    axis=alt.Axis(format="%", grid=False)),
            color=alt.Color(
                "side:N",
                scale=alt.Scale(domain=["home", "away"],
                                range=[MODEL_COLOR, "#8c8c8c"]),
                legend=None,
            ),
            tooltip=[alt.Tooltip("team:N"), alt.Tooltip("prob:Q", format=".1%")],
        )
        .properties(height=52)
    )
    st.altair_chart(chart, width='stretch')


def _prob_bars(probs: pd.DataFrame) -> None:
    """Cover probabilities as bars. Values arrive from the adapter untouched."""
    if not HAVE_ALT:
        st.dataframe(probs.style.format({"cover_prob": "{:.1%}"}),
                     width='stretch', hide_index=True)
        return
    base = alt.Chart(probs).encode(
        y=alt.Y("side:N", sort=None, title=None),
        x=alt.X("cover_prob:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", grid=False)),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=22, color=BAR_COLOR)
    # Right-aligned INSIDE the bar. Anchored to the bar's own end, a label sat exactly
    # where the 50% rule crosses and the rule painted over the leading digit -- 48.3% read
    # as 8.3%. Cover probabilities cluster near 50%, so there is always room inside.
    labels = base.mark_text(
        align="right", dx=-8, fontWeight="bold", color="white"
    ).encode(text=alt.Text("cover_prob:Q", format=".1%"))
    even = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(
        strokeDash=[4, 4], color="grey"
    ).encode(x="x:Q")
    # Rule first: last layer wins the z-order, and it must not sit on top of the numbers.
    st.altair_chart((even + bars + labels).properties(height=28 * len(probs) + 30),
                    width='stretch')


def _histogram(values: np.ndarray, title: str, rules: dict,
               key_numbers: "tuple | list" = (),
               weights: "np.ndarray | None" = None) -> None:
    """Histogram with the market and model lines drawn as vertical rules.

    `key_numbers` shades the football-specific margins (3, 7, ...) in a separate colour.
    Those spikes are the whole reason this model simulates drives rather than sampling a
    normal, and they were invisible in a flat blue histogram. Purely a colour encoding
    over bins that were already being drawn.
    """
    vals = np.asarray(values, dtype=float)
    lo, hi = np.percentile(vals, [0.5, 99.5])
    edges = np.arange(np.floor(lo), np.ceil(hi) + 1)
    counts, _ = np.histogram(vals, bins=edges, weights=weights)
    df = pd.DataFrame({"value": edges[:-1], "freq": counts / max(counts.sum(), 1)})

    keys = {abs(int(k)) for k in key_numbers}
    df["kind"] = np.where(
        df["value"].abs().astype(int).isin(keys) if keys else False,
        "key number", "other",
    )

    if not HAVE_ALT:
        st.caption(title)
        st.bar_chart(df.set_index("value")["freq"])
        return

    color = (
        alt.Color(
            "kind:N",
            scale=alt.Scale(domain=["other", "key number"],
                            range=[BAR_COLOR, KEY_COLOR]),
            # Bottom, not top-right: an in-plot legend sat over the tallest bars, which
            # are exactly the key-number spikes this chart exists to show.
            legend=alt.Legend(title=None, orient="bottom", direction="horizontal"),
        )
        if keys else alt.value(BAR_COLOR)
    )
    layers = [
        alt.Chart(df).mark_bar().encode(
            x=alt.X("value:Q", title=title),
            y=alt.Y("freq:Q", title="probability", axis=alt.Axis(format="%")),
            color=color,
            tooltip=[alt.Tooltip("value:Q"), alt.Tooltip("freq:Q", format=".2%")],
        )
    ]
    colors = {"market": MARKET_COLOR, "model": MODEL_COLOR}
    for name, x in rules.items():
        if x is None or pd.isna(x):
            continue
        rule_df = pd.DataFrame({"x": [float(x)], "label": [name]})
        layers.append(
            alt.Chart(rule_df)
            .mark_rule(color=colors.get(name, "black"), strokeWidth=2)
            .encode(x="x:Q", tooltip=[alt.Tooltip("label:N"),
                                      alt.Tooltip("x:Q", format="+.2f")])
        )
    st.altair_chart(alt.layer(*layers).properties(height=300), width='stretch')


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------

SECTIONS = ("Ratings", "Team", "Game explorer", "Diagnostics")


def _render_section(section: str, adapter, season: int, week: int) -> None:
    """Render exactly one section; Streamlit tabs eagerly execute every hidden body."""
    if section == "Ratings":
        tab_ratings(adapter, season, week)
    elif section == "Team":
        tab_team(adapter, season, week)
    elif section == "Game explorer":
        tab_game_explorer(adapter, season, week)
    elif section == "Diagnostics":
        tab_diagnostics(adapter)
    else:
        raise ValueError(f"unknown section: {section}")

def main() -> None:
    sport = resolve_sport()
    # shared/app.py owns page config when it hosts both sports; Streamlit errors if
    # set_page_config is called twice.
    if os.environ.get("VIEWER_EMBEDDED") != "1":
        st.set_page_config(page_title=f"{sport.upper()} model viewer", layout="wide")

    _inject_css()
    adapter = get_adapter(sport)
    # shared/app.py already prints the product title above the sport switch; a second H1
    # here stacked two titles on top of each other. When this file is run standalone
    # (nfl-model/app.py, ncaa-model/app.py) there is no outer title, so it still needs one.
    if os.environ.get("VIEWER_EMBEDDED") != "1":
        st.title(f"{adapter.profile.label} model viewer")

    if not adapter.available:
        st.error(
            f"The {adapter.profile.label} repo is not built yet "
            f"({adapter.profile.repo}). Build ingest and ratings first — per the amended "
            "build order the viewer comes third, so it is expected to run against a "
            "partially built repo."
        )
        return

    season = week = None
    with st.sidebar:
        # Controls first. The sidebar used to open with the repo directory name and the
        # market basis -- provenance, which matters, but not before the two selectors that
        # are the reason anyone opens the sidebar at all.
        st.markdown("### Period")
        periods = adapter.rating_periods()
        if periods is None or periods.empty:
            st.error("No rating periods found — run the ratings build.")
        else:
            default = adapter.default_period()
            seasons = sorted(periods["season"].unique(), reverse=True)
            s_idx = seasons.index(default[0]) if default and default[0] in seasons else 0
            season = st.selectbox("Season", seasons, index=s_idx)
            weeks = sorted(
                periods.loc[periods["season"] == season, "week"].unique(), reverse=True
            )
            w_idx = (
                weeks.index(default[1])
                if default and season == default[0] and default[1] in weeks
                else 0
            )
            week = st.selectbox("Week", weeks, index=w_idx)

        st.divider()
        with st.expander("Model details"):
            st.markdown(f"**Sport** · {adapter.profile.label}")
            st.markdown(
                f"**Market basis** · {adapter.profile.market_line_basis} line")
            st.markdown(f"**Source** · `{adapter.profile.repo.name}`")
            if adapter.profile.notes:
                st.caption(adapter.profile.notes)

    if season is None:
        return

    if hasattr(st, "segmented_control"):
        section = st.segmented_control(
            "Workspace section", SECTIONS, default=SECTIONS[0],
            key=f"section_{adapter.profile.key}", label_visibility="collapsed",
        ) or SECTIONS[0]
    else:
        section = st.radio(
            "Workspace section", SECTIONS, horizontal=True,
            key=f"section_{adapter.profile.key}", label_visibility="collapsed",
        )
    _status_bar(adapter, validate_gates=section in {"Game explorer", "Diagnostics"})
    _render_section(section, adapter, int(season), int(week))


if os.environ.get("VIEWER_EMBEDDED") != "1":
    main()
