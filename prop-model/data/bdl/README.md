# balldontlie NFL pull — 2026-08-22

Pulled once against a GOAT key, which was then cancelled. Not reproducible without paying
again, so treat these files as the archive rather than a cache.

## What is here

| file | rows | covers |
|---|---|---|
| `opening_odds.jsonl` | 5,052 | 2022-2025, 1,104 games, spreads + totals + moneylines **with prices**, from DraftKings / FanDuel / Caesars |
| `games.jsonl` | 1,412 (285/season) | 2022-2026, team abbreviations, dates, scores — the key that maps `game_id` to a matchup |
| `player_injuries.jsonl` | 334 | current snapshot — nfl-model has **no** live injury feed at all (nflverse clamps at 2024, `data/manual/availability.csv` does not exist) |
| `player_props.jsonl` | 800 | 2026 only, 3 games, DraftKings + FanDuel, both sides priced |

## READ THIS BEFORE USING `opening_odds.jsonl`

**These are NOT openers, despite the endpoint name.** Measured against kickoff:

| season | median hours before kickoff | p10 | p90 |
|---|---|---|---|
| 2022 | 30 | 13 | 33 |
| 2023 | 30 | 13 | 33 |
| 2024 | 29 | 13 | 32 |
| 2025 | 30 | 13 | 33 |

A genuine NFL opener is posted days to weeks ahead — usually right after the previous
week's games, about six days out. A median of 30 hours is a **day-before snapshot**.

This was caught by checking `opened_at` against kickoff rather than trusting the field name,
and it matters: an open-versus-close test run on this data showed the two numbers differing
by only 0.58 points with **1 of 182 games resolving differently**, which reads as "the
opener advantage disappeared after 2022" and is actually "these are two late snapshots".
The 2019-2021 Sportsbook Reviews archive in `scripts/fetch_sbr_odds.py` has real openers
(mean |open-close| 1.98) and remains the only genuine opener source here.

## What it IS good for

1. **A ~30-hour-out line with real prices, 2022-2025.** nflverse publishes closing lines
   only and no spread/total prices at all before 2016-era coverage. This is the first
   pre-kickoff snapshot with juice attached for the post-2022 era.
2. **It sits close to the W1 wind rule's 48-hour recording window**, so it is a reasonable
   proxy for "the number you could have taken when the tracker records".
3. **Independent confirmation of the wind edge.** Betting UNDER on outdoor games with wind
   >= 10 mph, 2022-2025, 189 games: **58.7 per 100 at this snapshot, 59.7 at the close** —
   against 58.3 measured separately from nflverse closing lines. Three sources, same answer.

## What is NOT here

**No historical player props.** Probed 2024 week 1 and 2025 weeks 1, 10 and 18: all returned
zero rows. Props appear only for 2026. The documented "most recently completed season" is
not populated for the NFL, so this key buys no prop backtest — the same limitation as the
free Odds API tier, and worth knowing before paying anyone for prop history.
