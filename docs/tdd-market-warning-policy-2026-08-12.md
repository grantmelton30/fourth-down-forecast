# Sparse market and warning policy — TDD evidence

## User journeys

- A viewer sees an explicit “No line posted yet” state when free sources have not priced
  a game, rather than mistaking a blank card for a low-risk comparison.
- A viewer sees the warning icon only for an extreme 20+ point model/market disagreement.
- The public market card uses the latest legitimate quote while historical evaluation
  continues to retain the opener separately.
- Free market sources refresh daily without rerunning expensive historical backtests.

## RED

`uv run pytest -q shared/tests/test_model_integrity.py shared/tests/test_ncaa_live_slate.py shared/tests/test_scheduled_refresh.py`

```text
4 failed, 14 passed
- 14.1 points was still classified extreme
- public slate used the opener instead of the latest quote
- no daily lightweight schedule existed
- the UI did not label missing lines and used every warning for the icon
```

RED checkpoints: `6650016`, `4dd71dd`.

## GREEN

The same focused target passed `19 passed`. Full isolated suites passed:

```text
shared: 62 passed
NFL: 52 passed, 1 skipped
NCAA: 18 passed, 6 skipped
```

The skipped tests are the existing external-historical-cache cases. Source audit found
that the free CFBD payload contained 888 scheduled games but only 53 with any offered
book line; the application cannot truthfully create the other 835 market quotes.
