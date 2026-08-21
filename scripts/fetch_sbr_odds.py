#!/usr/bin/env python
"""Historical NFL OPENING lines from Sportsbook Reviews Online.

    python scripts/fetch_sbr_odds.py --seasons 2019 2020 2021 --out data/manual/sbr_nfl.csv

WHY THIS EXISTS. nflverse publishes CLOSING lines only -- `spread_line` and `total_line`
are both closes, and there is no opener column for any season. That made "should we bet the
opener or the close" unanswerable on the NFL side, while NCAA could answer it from CFBD.
SBR publishes both, free, as an HTML table per season.

COVERAGE: 2007-08 through 2021-22 ONLY. The 2022-23 and later pages exist but carry no
table. That leaves the post-2022 efficiency regime (GATES.md) UNCOVERED, which is exactly
the era any current claim would need. Treat results from this source as evidence about
2007-2021 and not about the market as it trades today.

VALIDATION, run 2026-08-21 on 2019-2021: joined to nflverse on (date, both scores), 800 of
821 games matched, and SBR's CLOSE agrees with nflverse's independently sourced closing
total at r = 0.9943, 93.9% within half a point. Two sources that agree that closely on the
close are very unlikely to disagree about the open.

THE PARSING TRAP, and it is not obvious. SBR writes two rows per game and puts the game
TOTAL on one row and the favourite's SPREAD on the other -- so of the two values, the larger
is the total. The favourite can FLIP between open and close, which moves the total to the
other row. Resolving both columns off the open put a spread where a total belonged on 86 of
800 games, and produced a plausible-looking mean line move of 5.84 points (the true figure
is 1.98). Each column must be resolved independently.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import html
import numpy as np
import pandas as pd
import requests

BASE = "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nfl-odds-{a}-{b}"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
MAX_SEASON = 2021   # the archive stops here; later pages return 200 with no table


def fetch(season: int) -> str:
    url = BASE.format(a=season, b=str(season + 1)[2:])
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    return r.text


def parse(s: str, season: int) -> pd.DataFrame:
    m = re.search(r'<table.*?</table>', s, re.S|re.I)
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', m.group(0), re.S|re.I)
    recs = []
    for r in rows:
        cells = [html.unescape(re.sub(r'<[^>]+>', '', c)).strip()
                 for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', r, re.S|re.I)]
        if len(cells) >= 12 and cells[2] in ('V', 'H', 'N'):
            recs.append(cells)
    out = []
    for i in range(0, len(recs) - 1, 2):
        a, b = recs[i], recs[i+1]
        def num(x):
            x = x.strip().lower()
            if x in ('pk', 'p'): return 0.0
            try: return float(x)
            except ValueError: return np.nan
        ao, ac, bo, bc = num(a[9]), num(a[10]), num(b[9]), num(b[10])
        # SBR convention: the FAVOURITE's row carries the spread, the other carries the game
        # total, so of the two rows the larger value is the total. Critically the favourite
        # can FLIP between open and close, which moves the total to the other row -- so the
        # open and close columns must each be resolved on their own. Resolving both off the
        # open silently produced a spread where a total belonged on 86 of 800 games.
        def split(x, y):
            if np.isnan(x) or np.isnan(y): return np.nan, np.nan
            return (x, y) if x >= y else (y, x)
        tot_o, spr_o = split(ao, bo)
        tot_c, spr_c = split(ac, bc)
        if np.isnan(tot_o) and np.isnan(tot_c): continue
        md = re.sub(r'\D', '', a[0])
        if len(md) == 3: md = '0' + md
        if len(md) != 4: continue
        month = int(md[:2]); year = season if month >= 8 else season + 1
        out.append({
            'date': f'{year}-{md[:2]}-{md[2:]}', 'season': season,
            'away_team_sbr': a[3], 'home_team_sbr': b[3],
            'away_score': num(a[8]), 'home_score': num(b[8]),
            'total_open_sbr': tot_o, 'total_close_sbr': tot_c,
            'spread_open_sbr': spr_o, 'spread_close_sbr': spr_c,
        })
    return pd.DataFrame(out)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    late = [s for s in a.seasons if s > MAX_SEASON]
    if late:
        print(f"refusing {late}: the SBR archive stops at {MAX_SEASON}-"
              f"{str(MAX_SEASON+1)[2:]} and later pages return an empty table, which would "
              f"silently produce zero rows rather than an error.", file=sys.stderr)
        return 2
    out = []
    for season in a.seasons:
        df = parse(fetch(season), season)
        print(f"  {season}: {len(df)} games")
        out.append(df)
        time.sleep(1)   # a free archive, scraped once; do not hammer it
    frame = pd.concat(out, ignore_index=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(a.out, index=False)
    print(f"wrote {len(frame)} rows to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
