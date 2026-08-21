"""The two pre-registered forward-tracking rules, evaluated for one game.

    ncaa-model/PREREG.md  P1 -- totals, edge 0.5-6.0, restricted universe, either side
    nfl-model/PREREG.md   W1 -- totals, forecast wind >= 10 mph, outdoor, UNDER only

WHY THIS MODULE EXISTS. The rules are short but their preconditions are not, and no human
remembers them correctly at 11pm on a Saturday. Worse, they are easy to remember
OPTIMISTICALLY: the P5-vs-P5 exclusion is the single most valuable clause in P1 -- those
games lose 2.3 points below breakeven and would turn +39 units into -11 -- and it is
exactly the clause a person skips when a marquee game looks mispriced.

SINGLE SOURCE OF TRUTH, deliberately. The website and the trackers must agree about what
qualifies, and two implementations of the same rule diverge the moment one is edited. The
thresholds here are the same constants the registrations name, and `tests/test_tracked_
rules.py` asserts they still match.

IT SAYS "QUALIFIES", NOT "BET". `bets_allowed()` is False for both leagues -- six NFL gates
and five NCAA gates fail -- and neither registration authorises staking money. What this
answers is "does the declared rule fire on this game", which is a question about a tracking
protocol, not a recommendation. The distinction is not pedantry: it is the only reason the
forward record will be worth reading in December.

THE REASON STRING IS THE POINT. A bare no teaches nothing and gets overridden. "No -- P5 vs
P5, outside the tracked universe" teaches the rule every time it is read.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

# --- P1 (NCAA), from ncaa-model/PREREG.md ---------------------------------------------
P1_MIN_EDGE = 0.5
P1_MAX_EDGE = 6.0

# --- W1 (NFL), from nfl-model/PREREG.md -----------------------------------------------
W1_MIN_WIND_MPH = 10.0
W1_MAX_WIND_MPH = 40.0

RULE_FOR_LEAGUE = {"ncaa": "P1", "nfl": "W1"}


@dataclass(frozen=True)
class RuleStatus:
    """Whether the league's tracked rule fires, and in plain words why or why not."""
    rule: str                 # "P1" | "W1"
    qualifies: bool
    side: "str | None"        # "OVER" | "UNDER" | None
    reason: str               # human-readable, always populated
    market: str = "total"

    @property
    def label(self) -> str:
        """Short badge text for a table cell."""
        return f"{self.rule} · {self.side}" if self.qualifies else "—"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["label"] = self.label
        return d


def _num(value):
    """None for anything not a real number, including NaN, without importing pandas."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def evaluate_p1(*, model_total, market_total, restricted) -> RuleStatus:
    """P1: NCAA totals, restricted universe, model-market gap in [0.5, 6.0).

    Bets BOTH ways -- over if the model is above the market, under if below -- unlike W1,
    which is one-sided because its mechanism is.
    """
    R = "P1"
    # Universe first: it is the clause most likely to be rationalised away, and checking it
    # before the edge means a tempting number never gets shown next to a qualifying gap.
    if restricted is None:
        return RuleStatus(R, False, None, "No — universe unknown (conference data missing)")
    if not bool(restricted):
        return RuleStatus(
            R, False, None,
            "No — outside the tracked universe (P5-vs-P5 or non-FBS). "
            "P5-vs-P5 measured 50.1 per 100, below breakeven.")

    m, k = _num(model_total), _num(market_total)
    if m is None:
        return RuleStatus(R, False, None, "No — no model total")
    if k is None:
        return RuleStatus(R, False, None, "No — no market total posted yet")

    edge = m - k
    size = abs(edge)
    if size < P1_MIN_EDGE:
        return RuleStatus(
            R, False, None,
            f"No — model agrees with the market ({size:.1f} pts, "
            f"under the {P1_MIN_EDGE} minimum)")
    if size >= P1_MAX_EDGE:
        return RuleStatus(
            R, False, None,
            f"No — gap of {size:.1f} pts is at or above the {P1_MAX_EDGE} cap")
    side = "OVER" if edge > 0 else "UNDER"
    return RuleStatus(
        R, True, side,
        f"Yes — model {'above' if edge > 0 else 'below'} the market by {size:.1f} pts")


def evaluate_w1(*, wind_mph, roof, venue_roof=None) -> RuleStatus:
    """W1: NFL totals, outdoor, forecast wind >= 10 mph, always UNDER.

    `roof` is nflverse's per-game state and is null for every unplayed game. `venue_roof`
    is the stadium's structural type, used only to tell a dome from a retractable whose
    game-day state has not been published -- the latter is UNKNOWN, not open.

    Consults no model output at all. That is deliberate: this model's disagreements with the
    NFL market are measurably worse than useless since 2023 (GATES.md), so the rule bets a
    physical fact instead.
    """
    R = "W1"
    roof_s = str(roof).strip().lower() if isinstance(roof, str) else None
    if roof_s in ("dome", "closed"):
        return RuleStatus(R, False, None, "No — indoors, wind cannot matter")
    vr = str(venue_roof).strip().lower() if isinstance(venue_roof, str) else None
    if roof_s is None and vr == "dome":
        return RuleStatus(R, False, None, "No — indoors, wind cannot matter")
    if roof_s is None and vr == "retractable":
        return RuleStatus(
            R, False, None,
            "No — retractable roof, game-day state not published. Unknown is not open.")

    w = _num(wind_mph)
    if w is None:
        return RuleStatus(R, False, None, "No — no wind forecast available yet")
    if w > W1_MAX_WIND_MPH:
        return RuleStatus(
            R, False, None,
            f"No — {w:.0f} mph forecast is implausible and is rejected, not treated as calm")
    if w < W1_MIN_WIND_MPH:
        return RuleStatus(
            R, False, None,
            f"No — wind {w:.0f} mph, under the {W1_MIN_WIND_MPH:.0f} mph trigger")
    return RuleStatus(R, True, "UNDER", f"Yes — wind {w:.0f} mph, at or over the trigger")


def evaluate(league: str, **kw) -> RuleStatus:
    """Dispatch to the tracked rule for `league`. Unknown leagues have no rule."""
    lg = str(league).lower()
    if lg == "ncaa":
        return evaluate_p1(
            model_total=kw.get("model_total"), market_total=kw.get("market_total"),
            restricted=kw.get("restricted"))
    if lg == "nfl":
        return evaluate_w1(
            wind_mph=kw.get("wind_mph"), roof=kw.get("roof"),
            venue_roof=kw.get("venue_roof"))
    return RuleStatus("—", False, None, f"No tracked rule for league {league!r}")


__all__ = ["RuleStatus", "evaluate", "evaluate_p1", "evaluate_w1",
           "P1_MIN_EDGE", "P1_MAX_EDGE", "W1_MIN_WIND_MPH", "W1_MAX_WIND_MPH"]
