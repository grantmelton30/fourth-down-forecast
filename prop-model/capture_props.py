#!/usr/bin/env python
"""Capture NFL player prop quotes into an append-only ledger.

    python capture_props.py verify                 # what does this key actually allow?
    python capture_props.py capture                # counting markets, all upcoming events
    python capture_props.py capture --markets player_receptions --max-events 4
    python capture_props.py report                 # what is in the ledger

WHY THIS RUNS BEFORE ANY MODEL EXISTS. A quote nobody wrote down cannot be recovered. The
same argument `scripts/collect_market_quotes.py` already makes for game lines applies harder
to props, where no free historical archive exists at any price — the vendor's own history
starts 2023-05-03 and is paid-only. Every week not captured now is a week that can never be
part of a backtest.

BOTH PRICES, EVERY BOOK, NO AVERAGING. Two external audits found the same defect in the
existing build: lines recorded without prices, and a two-book median that manufactures
quarter-point numbers no book offers. A row here is one quote from one book that somebody
could actually have taken.

CREDIT DISCIPLINE. The free tier is 500 credits a month and live event odds cost
`markets x regions` PER EVENT. A full NFL week of one market is ~16 credits; three markets
~48. `--max-events` and the pre-flight estimate exist so a run cannot quietly spend a
month's allowance. Usage is read from the response headers, not estimated.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from src.odds_api import (COUNT_MARKETS, OddsApiError, Usage,  # noqa: E402
                              fetch_event_odds, flatten_props, list_events, verify_access)
    from src.prop_ledger import PropLedger  # noqa: E402
except ModuleNotFoundError as exc:
    # Running under the system interpreter instead of the repo's venv is the first thing
    # that goes wrong here, and a bare ImportError traceback does not say so. This is a
    # weekly command; it should tell you the fix rather than the symptom.
    _venv = Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "python.exe"
    if not _venv.exists():
        _venv = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
    sys.exit(
        f"{exc}\n\n"
        f"That usually means this ran under the system Python rather than the repo's\n"
        f"virtual environment, which is where the dependencies live. Use either:\n\n"
        f"    {_venv} capture_props.py {' '.join(sys.argv[1:]) or 'verify'}\n"
        f"    uv run python prop-model/capture_props.py {' '.join(sys.argv[1:]) or 'verify'}"
        f"   (from the repo root)\n"
    )

LEDGER = Path(__file__).resolve().parent / "data" / "prop_quotes.jsonl"


def cmd_verify(args) -> int:
    try:
        report = verify_access()
    except OddsApiError as exc:
        print(f"FAILED: {exc}")
        return 1
    print(json.dumps(report, indent=2))
    ok = report.get("nfl_listed") and any(
        m.get("available") for m in report.get("markets", {}).values())
    print("\n" + ("PLAYER PROPS ARE AVAILABLE on this key."
                  if ok else
                  "Player props are NOT available on this key -- do not build on it."))
    return 0 if ok else 1


def cmd_capture(args) -> int:
    markets = args.markets or list(COUNT_MARKETS)
    usage = Usage()
    try:
        events = list_events(usage=usage)
    except OddsApiError as exc:
        print(f"FAILED to list events: {exc}")
        return 1
    if args.max_events:
        events = events[: args.max_events]
    if not events:
        print("no upcoming NFL events -- nothing to capture")
        return 0

    estimate = len(events) * len(markets)
    print(f"{len(events)} event(s) x {len(markets)} market(s) "
          f"= ~{estimate} credits (1 per event per market)")
    if usage.remaining is not None and estimate > usage.remaining:
        print(f"REFUSING: estimate {estimate} exceeds remaining quota {usage.remaining}. "
              f"Narrow with --markets or --max-events.")
        return 2

    ledger = PropLedger(LEDGER)
    captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows, failures = [], []
    for event in events:
        try:
            payload = fetch_event_odds(event["id"], markets, usage=usage)
        except OddsApiError as exc:
            # Recorded, not swallowed: a gap in the archive must be attributable to an
            # outage rather than to a market that never opened.
            failures.append({"event_id": event["id"], "error": str(exc)[:200]})
            print(f"  {event.get('away_team')} at {event.get('home_team')}: FAILED {exc}")
            continue
        got = flatten_props(payload, captured_at=captured_at)
        rows.extend(got)
        print(f"  {event.get('away_team')} at {event.get('home_team')}: {len(got)} quotes")

    added = ledger.append(rows)
    print(f"\nappended {added} new quote(s) ({len(rows) - added} already present)")
    print(f"usage: {usage}")
    if failures:
        print(f"{len(failures)} event(s) failed -- see above. These are gaps in the archive.")
        return 1
    return 0


def cmd_report(args) -> int:
    ledger = PropLedger(LEDGER)
    print(json.dumps(ledger.summary(), indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify", help="ask the API what this key allows")
    cap = sub.add_parser("capture", help="capture prop quotes for upcoming events")
    cap.add_argument("--markets", nargs="+")
    cap.add_argument("--max-events", type=int)
    sub.add_parser("report", help="what is in the ledger")
    args = p.parse_args()
    return {"verify": cmd_verify, "capture": cmd_capture, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
