# NCAA live-slate publication — TDD evidence

## User journey

As a viewer, I want the current NCAA week projected with the same independent, market,
calibrated, difference, and data-quality contract as NFL, so NCAA does not disappear when
a processed historical cache predates the live season.

## RED / GREEN evidence

| Guarantee | Test | RED evidence | GREEN evidence |
|---|---|---|---|
| Cached 2026 schedule and line payloads are merged over a stale historical market artifact | `test_ncaa_market_adds_live_raw_cache_to_stale_history` | Live-season row absent; `.iloc[0]` failed | Live row present with normalized +4.5 home spread and 51.5 total |
| NCAA simulation requests the configured live season | `test_ncaa_schedule_requests_configured_live_season` | Requested seasons ended at 2025 | Requested seasons end at 2026 |
| An expired live-cache TTL without a configured API key falls back to the existing cache instead of removing NCAA | `test_ncaa_schedule_falls_back_to_cached_live_payload_without_api_key` | Missing-key `RuntimeError` aborted the schedule | Cached 2026 game loaded with parsed kickoff |

Focused command:

```text
PYTHONPATH="$PWD/shared:$PWD/ncaa-model" PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
-p no:cacheprovider -q shared/tests/test_ncaa_live_slate.py
```

Final focused result: `3 passed`.

Full isolated repository result (`.venv/bin/python run_tests.py`):

```text
shared: 40 passed
NFL:    52 passed, 1 skipped
NCAA:   11 passed, 6 skipped
total:  103 passed, 7 skipped, 0 failed
```

## Publication verification

The cached free-source snapshot produced 99 NCAA 2026 Week 1 independent projections.
Forty-nine have both a market spread and total and therefore a calibrated comparison.
All 99 retain the `limited data` disclosure because the required three-source consensus
and confirmed QB/coordinator continuity inputs are not present. No quality gate was
weakened to make the slate appear.

## Known gap

The repository can publish from the last valid cache without a secret. Refreshing CFBD
schedule and market payloads after their TTL requires a free `CFBD_API_KEY` in the runtime
environment. A stale snapshot is disclosed through its prediction data cutoff and does
not authorize picks.
