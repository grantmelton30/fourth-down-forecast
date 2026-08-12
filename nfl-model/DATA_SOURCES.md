# DATA SOURCES — APPENDIX FOR BOTH PLAYBOOKS

Every source, where to sign up, what it costs, what it covers, and what to do when it
breaks. Both playbooks reference this file. Read it before writing any ingest code.

Attach this file alongside whichever playbook you paste into Claude Code.

---

## THE SHORT VERSION

| Need | NFL source | NCAA source | Cost |
|---|---|---|---|
| Play-by-play | nflverse via `nflreadpy` | CFBD `PlaysApi` | Free |
| Drives | derive from PBP | CFBD `DrivesApi` | Free |
| Schedules & results | nflverse `load_schedules()` | CFBD `GamesApi` | Free |
| **Historical closing lines** | nflverse `load_schedules()` | CFBD `BettingApi` | **Free** |
| Live lines this week | The Odds API *or* manual CSV | CFBD `BettingApi` *or* manual CSV | Free tier / free |
| Injuries & depth charts | nflverse `load_injuries()`, `load_depth_charts()` | **None exists — manual** | Free / n/a |
| Weather forecast | Open-Meteo | Open-Meteo | Free, no key |
| Historical weather | nflverse `schedules.temp/wind` + Open-Meteo archive | Open-Meteo archive | Free |
| Stadium coordinates | hardcoded table (below) | CFBD `VenuesApi` | Free |
| Returning production | n/a | CFBD `PlayersApi` | Free |
| QB valuation | **nfeloqb `qb_elos.csv`** | none exists | Free |
| Blend third term | **nfelo model** (open source) | CFBD SP+ | Free |
| Baseline model to beat | nfelo / your own Elo | CFBD `RatingsApi` (SP+) | Free |

The single most important line in this table: **historical closing lines are free for both
sports.** You do not need a paid odds API to backtest. That is the thing most people
assume they have to pay for.

---

## 1. NFL — nflverse

**Package:** `nflreadpy` — https://nflreadpy.nflverse.com
**Source repo:** https://github.com/nflverse/nflreadpy
**Data repo:** https://github.com/nflverse/nflverse-data
**Function reference:** https://nflreadpy.nflverse.com/api/load_functions/
**Data dictionaries:** https://nflreadr.nflverse.com/articles/ (the R docs are the
canonical dictionaries; the Python package loads the same files)

Install: `pip install nflreadpy`. No API key, no signup, no rate limit. It reads parquet
files published to GitHub releases.

**`nfl_data_py` is deprecated.** Its README says so explicitly and directs users to
nflreadpy. If Claude Code reaches for `nfl_data_py` out of training habit, stop it.

### Functions this build uses

```python
import nflreadpy as nfl

nfl.load_pbp(seasons)             # play-by-play, 1999+
nfl.load_schedules(seasons)       # games, results, AND betting lines
nfl.load_players()                # player master, all IDs
nfl.load_rosters_weekly(seasons)  # week-by-week roster
nfl.load_depth_charts(seasons)    # depth charts  <-- QB module
nfl.load_injuries(seasons)        # injury report + practice participation, 2009+  <-- QB module
nfl.load_snap_counts(seasons)     # PFR snap counts, 2012+
nfl.load_participation(seasons)   # historical only, do not rely on it for current season
nfl.load_ftn_charting(seasons)    # 2022+, optional, CC-BY-SA license (see §8)
```

All return **Polars**. Call `.to_pandas()` at the ingest boundary.

### `load_schedules()` is the backbone

Columns you need and what they mean:

- `spread_line` — **the closing spread. Positive = home favored.** Aligned with `result`.
- `total_line` — closing total.
- `home_moneyline`, `away_moneyline` — closing moneylines.
- `result` — `home_score - away_score`. `total` — combined score.
- `roof` — `outdoors` / `dome` / `closed` / `open`. `surface` — turf type.
- `temp`, `wind` — **populated for completed outdoor games only.** Null for upcoming
  games and null for domes. This is why you need Open-Meteo (§4).
- `home_rest`, `away_rest` — days since previous game.
- `div_game`, `location` (`Home` / `Neutral`), `overtime`, `game_type`.
- `gameday`, `gametime`, `weekday`.

Maintained by Lee Sharpe at https://github.com/nflverse/nfldata — the dataset docs are at
https://github.com/nflverse/nfldata/blob/master/DATASETS.md. Read that page once; it is
the authoritative column reference.

### Fallback if the package breaks

The package is a thin wrapper over static parquet files. If `nflreadpy` errors, read the
files directly:

```
https://github.com/nflverse/nflverse-data/releases/tag/pbp
https://github.com/nflverse/nflverse-data/releases/tag/rosters
```

and for schedules, the raw CSV that never moves:

```
https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv
```

Write the ingest layer so that swapping to a direct URL read is a one-function change.
Update status is published at https://github.com/nflverse/nflverse-data — check it if
current-week data looks stale.

---

## 2. NCAA — CollegeFootballData

**Site & signup:** https://collegefootballdata.com
**Get a key:** https://collegefootballdata.com/key — email address, key arrives by email.
**API docs:** https://apinext.collegefootballdata.com
**Python package:** `pip install cfbd` — https://github.com/CFBD/cfbd-python
**Blog with worked examples:** https://blog.collegefootballdata.com

Free tier is **1,000 calls per month**. Key auth is enforced on all requests. Paid tiers
via Patreon add weather, advanced metrics, GraphQL, and higher limits.

### The endpoints this build uses

```python
cfbd.GamesApi        .get_games(year, season_type)        # schedule + results
cfbd.DrivesApi       .get_drives(year)                    # drive-level, cleaner than deriving
cfbd.PlaysApi        .get_plays(year, week)               # paginate by week
cfbd.BettingApi      .get_lines(year, week)               # spreads, totals, open + current
cfbd.PlayersApi      .get_returning_production(year)      # the key preseason input
cfbd.RatingsApi      .get_sp(year)                        # SP+, BASELINE ONLY
cfbd.RecruitingApi   .get_recruiting_teams(year)          # 2% prior weight
cfbd.TeamsApi        .get_fbs_teams(year)                 # team list + conference
cfbd.VenuesApi       .get_venues()                        # lat/lon/elevation/dome  <-- once, ever
```

### Two things that will bite you

**Spread sign convention is opposite to nflverse.** CFBD quotes spreads from the home
team's perspective with negative meaning home favored. nflverse's `spread_line` is
positive when home is favored. Normalize at ingest to one convention — use nflverse's —
and write a test that asserts the sign on a known historical blowout. A sign-flipped
model produces entirely plausible-looking numbers while being exactly backwards.

**Weather is behind the Patreon tier.** Do not build against `getWeather`. Use Open-Meteo
plus the venue coordinates from `VenuesApi` instead — it is free and better anyway,
because the archive lets you backfill historical weather for the backtest.

### Budget discipline

Pull whole seasons, never weeks, except for `get_plays` which must paginate. Historical
seasons are immutable — cache them to parquet once and never refetch. Roughly 25 calls
buys you an entire season. Seven seasons of history is about 175 calls, one time, out of
1,000 per month. The in-season weekly refresh should cost 3-5 calls.

The `BudgetedCFBD` wrapper in the playbook exists to enforce this. Do not bypass it.

---

## 3. LIVE LINES FOR THE CURRENT WEEK

You need this only to price *this week's* slate. The backtest does not need it.

### Option A — CFBD betting endpoint (NCAA, and it is free)

`BettingApi.get_lines(year, week)` returns current lines by provider. Use this for
college. It costs one call.

### Option B — The Odds API (NFL)

**Signup:** https://the-odds-api.com/#get-access
**Docs:** https://the-odds-api.com/liveapi/guides/v4/
**NFL page:** https://the-odds-api.com/sports-odds-data/nfl-odds.html

```
GET https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds
    ?regions=us&markets=h2h,spreads,totals&oddsFormat=american&apiKey=YOUR_KEY
```

**A caution on tiers.** There is more than one product marketing itself under this name,
and their free tiers differ — one documents a free tier limited to NBA and MLB moneylines
only, which would be useless here. Verify what your key actually returns for
`americanfootball_nfl` before building anything on top of it. Call `/v4/sports/` first
and confirm NFL appears in the response. If it does not, fall back to Option C
immediately rather than paying to find out.

Each request costs quota equal to `markets × regions`. Requesting three markets in one
US-region call costs 3, not 1 — so one weekly pull is cheap. Do not poll.

### Option C — manual CSV, and this is genuinely fine

`data/manual/this_week_lines.csv`:

```csv
away_team,home_team,spread_home,total,spread_price_home,spread_price_away,total_price_over,total_price_under
```

The playbook generates this pre-filled with matchups and blank numbers. You type in the
numbers off your betting app in about two minutes for NFL, ten for a college Saturday.

This is not a degraded path. You are shopping the number on your app anyway, so the
numbers you type are the ones you can actually get — which is more accurate than a
consensus feed that includes books you do not have accounts at. Build the API path, but
do not treat the manual path as a compromise.

---

## 4. WEATHER — Open-Meteo

**Site:** https://open-meteo.com
**Docs:** https://open-meteo.com/en/docs
**Archive docs:** https://open-meteo.com/en/docs/historical-weather-api

No API key. No signup. Free for non-commercial use up to 10,000 calls per day.
Attribution required under CC BY 4.0 — put a line in your README.

**Forecast** (upcoming games, up to 16 days out):

```
https://api.open-meteo.com/v1/forecast
  ?latitude=42.09&longitude=-71.26
  &hourly=temperature_2m,wind_speed_10m,precipitation
  &temperature_unit=fahrenheit&wind_speed_unit=mph
  &timezone=America/New_York
```

**Archive** (backfilling historical weather for the backtest, data back to 1940):

```
https://archive-api.open-meteo.com/v1/archive
  ?latitude=42.09&longitude=-71.26&start_date=2019-09-08&end_date=2019-09-08
  &hourly=temperature_2m,wind_speed_10m
```

Set units explicitly — the default is Celsius and km/h, and a wind threshold of 15
"mph" silently evaluated against km/h will suppress totals on every windless afternoon.
Pull the hourly value nearest kickoff, not the daily mean.

**Skip the call entirely for indoor venues.** Check `roof` (NFL) or the dome flag from
`VenuesApi` (NCAA) first. Roughly a third of NFL games are indoors and a weather call for
them is pure waste.

For the NFL backtest specifically, `schedules.temp` and `schedules.wind` already carry
observed conditions for completed outdoor games — use those and only call the Open-Meteo
archive to fill nulls.

---

## 5. INJURIES AND PLAYER AVAILABILITY

This is the honest gap in the whole build, and the two sports are in completely different
situations.

### NFL — partially solved

`nfl.load_injuries()` gives the official weekly injury report back to 2009: player, team,
week, `report_status` (Out / Doubtful / Questionable), `practice_status`, and body part.
`nfl.load_depth_charts()` gives weekly depth chart position.

Use them together in the QB module:

1. Pull the depth chart for the upcoming week, take the QB1.
2. Cross-reference against the injury report. If QB1 is `Out` or `Doubtful`, promote QB2.
3. Write the result into `data/manual/qb_starters.csv` as a **pre-filled default**.
4. You override it by hand when news breaks.

Keep the manual override. Injury reports are strategically vague, "Questionable" resolves
both ways, and beat reporters know before the report does. The automation gets you 90% of
the way and saves you from the two minutes of tedium; the override handles the 10% where
the money is.

### NCAA — not solved, and be honest about it

There is no reliable free college football injury feed. The NCAA has no mandatory injury
report, availability news lives in local beat reporting and Twitter, and the transfer
portal and opt-outs create availability questions that no structured source covers.

The playbook's response is to make this visible rather than pretend otherwise:

- `data/manual/qb_starters.csv` and `data/manual/bowl_opt_outs.csv` are manual, full stop.
- Bowl games with an unfilled opt-out file emit **NO BET**, not a default of zero opt-outs.
- `run_week.py` warns if a manual file has not been touched in 14 days.

Practically: check the depth chart and availability news for the games your model already
flagged as bets, not for all 60 games. That is five to eight games a week, which is ten
minutes. Doing it in that order — model first, research second — also keeps you from
talking yourself into a game the model does not like.

---

## 6. STADIUM COORDINATES

### NCAA — free from the API

`VenuesApi.get_venues()` returns every venue with latitude, longitude, elevation, capacity,
dome flag, and city/state. **Call this once, cache it permanently, never call it again.**
Elevation is what drives the altitude adjustment.

### NFL — hardcode it

32 teams. Do not scrape, do not call an API, do not add a geocoding dependency for 32 rows
that change once a decade. Write `src/stadiums.py` with a literal dict:

```python
STADIUMS = {
    "NE": {"lat": 42.0909, "lon": -71.2643, "tz": "America/New_York", "roof": "outdoors", "elev_ft": 289},
    "DEN": {"lat": 39.7439, "lon": -105.0201, "tz": "America/Denver", "roof": "outdoors", "elev_ft": 5280},
    # ... all 32
}
```

Source the coordinates from Wikipedia stadium pages or Google Maps; they are public facts,
not a dataset. Cross-check the `roof` value against `schedules.roof` for a recent game per
team, since several teams have moved or changed venues (LV, LA×2, BUF, TEN). Handle
international games — London, Munich, Mexico City, São Paulo — by reading `location` from
schedules and treating them as neutral sites with zero HFA rather than trying to model
them.

---

## 7. BASELINE MODELS TO MEASURE AGAINST

You need something to beat, or the backtest numbers are meaningless in isolation.

- **NCAA:** SP+ via `RatingsApi.get_sp(year)`. Bill Connelly's model, published at ESPN.
  This is your benchmark, not your input. The gate is "within 0.5 points of SP+ RMSE."
- **NFL:** nfelo publishes power ratings and margin methodology at
  https://www.nfeloapp.com/nfl-power-ratings/ — useful reading for how a good public model
  handles margin distributions and home field. Build your own simple Elo as the
  in-repo baseline.
- **Both:** the market itself. The closing line is the benchmark that matters most.

Do not feed any of these into the model. A model that consumes SP+ is a wrapper around
SP+, and its only edge is the lag between SP+ publishing and the market absorbing it,
which is approximately zero.

---

## 7b. NFELO — NFL ONLY

**Site:** https://www.nfeloapp.com
**Main model repo:** https://github.com/greerreNFL/nfelo
**QB model repo:** https://github.com/greerreNFL/nfeloqb
**Data loader:** `pip install nfelodcm` — https://github.com/greerreNFL/nfelodcm
**About / methodology:** https://www.nfeloapp.com/about/

nfelo is an open-source NFL Elo model. All code is public, the backtest runs to 2009, and
picks are independently tracked by PredictionTracker.com. It is built on nflverse, so it
shares your data foundation.

### The one file you should pull first

```
https://raw.githubusercontent.com/greerreNFL/nfeloqb/main/qb_elos.csv
```

`nfeloqb` recreates and maintains 538's QB Elo model in 538's original CSV schema, with
documented improvements to seasonal regression, rookie initial values, and opponent
defensive adjustment. **Updated every Tuesday and Thursday morning during the season.**
Seasons through 2022 use 538's original model; 2023 onward uses nfeloqb's.

Columns that matter: `season`, `date`, `team1`, `team2`, `qb1`, `qb2`,
`qb1_value_pre`, `qb2_value_pre`, `qb1_adj`, `qb2_adj`, `qbelo1_pre`, `qbelo2_pre`.

`qb1_adj` / `qb2_adj` are a **points-scale QB adjustment** — exactly the quantity the NFL
playbook's QB module needs, from a maintained source with a decade of history. Plain HTTP
GET, cache to parquet, refresh weekly. No key, no package.

### Getting historical nfelo spreads for the blend

Harder than the QB file, and this is the one piece of the build that may not land cleanly.
The main model is open source and runnable locally, which is the reliable route to a
historical `nfelo_spread` series; `nfelodcm` is the author's data-loading layer for
assembling inputs. Budget real time for it.

**Degrade gracefully.** If running the full model locally proves impractical, set
`b_nfelo` to zero, run the two-term blend, and note it in `DECISIONS.md`. The build must
not block on this. The QB file above is the high-value, low-effort half of the nfelo
integration; the spread series is the low-value, high-effort half.

### The rule that keeps this honest

**Blend, never ratings.** nfelo's spread enters only as a residual term
`(nfelo_spread - market_spread)` in the blend regression, where OLS decides whether it
earns weight. It never touches the ridge. A model that consumes nfelo inside its ratings
is a wrapper around nfelo, and its only edge is the publish-to-market lag, which for a
public model is approximately zero.

nfelo also regresses its own output to market spreads by design — which is why the
residual form is mandatory rather than stylistic. Regress on levels and the collinearity
makes every coefficient meaningless.

### No NCAA equivalent

nfelo is NFL-only, and no totals model. The college analog is SP+ from CFBD, used the
same way: residual term in the blend, baseline in the backtest, never an input to ratings.

### Attribution

Credit nfelo in your README. Free, open source, MIT-spirited, and it saves you weeks of
work on the QB problem alone.

---

## 8. LICENSING AND ATTRIBUTION

Put these in your README. They cost you one paragraph and they are the terms you agreed to.

- **nflverse data** — CC-BY 4.0. Attribute nflverse.
- **FTN charting data** (only if you use `load_ftn_charting`) — CC-BY-SA 4.0, and
  attribution must name **FTN Data via nflverse**. The share-alike term is stricter than
  the rest of nflverse, which is why the playbook marks it optional. Skip it unless you
  have a specific need.
- **CollegeFootballData** — free tier for personal use; credit CollegeFootballData.com.
  Do not redistribute bulk data.
- **Open-Meteo** — CC-BY 4.0, attribution required, free tier is non-commercial.
- **The Odds API** — per their terms; do not redistribute odds.
- **nfelo / nfeloqb** — open source on GitHub. Credit nfelo (nfeloapp.com).

All fine for a personal model. All would need revisiting if you ever sold access to it.

---

## 9. WHEN A SOURCE BREAKS — RUNBOOK

Every ingest function should fail loudly with the specific cause, never silently return
an empty frame. An empty frame propagates into ratings as "this team has no data," which
the ridge model dutifully shrinks to league average, which produces a plausible-looking
line built on nothing.

| Symptom | Likely cause | Do this |
|---|---|---|
| Current-week PBP is missing | nflverse updates on a schedule, not instantly after games | Check https://github.com/nflverse/nflverse-data — it publishes update times. Wait, do not rebuild. |
| `nflreadpy` import or fetch errors | Package or release-asset change | Read the parquet URLs directly (§1 fallback) |
| CFBD returns 401 | Key missing, expired, or not in the env | Check `.env`, confirm `CFBD_API_KEY` is loaded |
| CFBD returns 429 or budget error | Monthly cap | Check `data/api_budget.json`. Work from cache until the month rolls. This is why the cache is permanent. |
| Odds API returns 403 on NFL | Your tier does not include NFL | Fall back to the manual CSV. Do not upgrade to diagnose. |
| A team's rating swings 15+ points in a week | Team abbreviation change broke a join, or garbage-time filter failed | Check `normalize_team()` coverage and the % of plays dropped by the filter |
| Every home team looks like value | HFA constant is wrong, or spread sign is flipped | Assert on a known game. This is the single most common silent bug. |
| Backtest looks great, live CLV is negative | Backtest is grading against stale or closing lines you could not actually get | Regrade against opening lines only |

Add an `ingest_health()` function that runs before every weekly refresh and prints: rows
per table, max date per table, count of teams with fewer than 3 games, and percentage of
plays dropped as garbage time. Read it every week. It catches most of the table above
before it reaches the bet sheet.
