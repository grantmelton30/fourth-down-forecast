# Free website deployment

The public site is a static read-only client. It cannot import model code, fit a model,
or mutate a projection. `scripts/publish_ledger.py` converts the append-only
`data/predictions.jsonl` ledger into `web/api/v1/predictions.json`; the prebuilt
`web/api/v1/explorer.json` supplies league rankings and schedules. The browser never
loads model caches or fits projections, preserving both fast startup and published values.

## Local preview

From the project root:

```bash
./.venv/bin/python scripts/build_static_site.py
./.venv/bin/python -m http.server 8765 --directory web
```

Open <http://localhost:8765>. This static site should open immediately; it does not wait
for simulations, pandas imports, CFBD calls, or Streamlit startup.

## GitHub

Create an empty GitHub repository, then initialize and push this folder. The `integrity`
workflow runs the three Python suites in separate processes and smoke-tests the website.
The `publish-ledger` workflow rebuilds the JSON twice weekly and commits only when it
changed. It uses GitHub Actions rather than a Render cron job so the initial setup has no
required monthly charge.

## Render (free static site)

1. In Render choose **New > Blueprint** and connect the GitHub repository.
2. Render reads `render.yaml`; approve the `fourth-down-forecast` static service.
3. No environment variables are required to serve the site.
4. Every push to the selected branch triggers a deploy.

Do not put `CFBD_API_KEY` or other source credentials in browser code. If a future refresh
workflow needs CFBD, store the key as a GitHub Actions secret and run the Python source
adapter there. The generated JSON remains the only public artifact.

## Publishing actual forecasts

The site intentionally begins empty. A slate should be published only after the game
pipeline creates `PredictionRecord` objects with:

- an information cutoff strictly before kickoff;
- the immutable independent football forecast;
- dated market evidence (3+ providers to claim consensus);
- an expanding-season calibrated forecast, if available;
- unavailable feature disclosures and the model version.

Append each record through `PredictionLedger.append`, rerun `scripts/publish_ledger.py`,
and push. Never hand-edit the public JSON to make a projection look better.

For the next priced week, the supported command is:

```bash
./.venv/bin/python scripts/build_slate.py --league all
./.venv/bin/python scripts/publish_ledger.py
```

Copy `data/manual/market_lines.csv.example` to `data/manual/market_lines.csv` and enter
dated free line observations. Fewer than three independent providers is displayed as a
single-line baseline, never as consensus.
