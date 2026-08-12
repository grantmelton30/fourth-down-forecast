"""Budget-aware CFBD wrapper. This module exists solely to keep the build inside the free
tier (1,000 calls/month).

Every call goes through `BudgetedCFBD.call()`: check the permanent cache first, and only on
a miss check the budget, hit the network, write the cache, and increment a counter that
persists to disk. Historical seasons are immutable, so once cached they are never refetched
and a cold backtest costs its calls exactly once.

Uses the REST API directly rather than the generated `cfbd` package: the endpoints needed
here are simple GETs, and a thin client keeps budget accounting in one obvious place
instead of behind an auto-generated layer.
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from .config import BUDGET_PATH, CACHE_DIR, Config, load_api_key

BASE_URL = "https://api.collegefootballdata.com"


class APIBudgetExceeded(RuntimeError):
    """Raised instead of silently making a call that would blow the monthly cap."""


class BudgetedCFBD:
    def __init__(self, cfg: Config, budget_path: Path = BUDGET_PATH):
        self.cfg = cfg
        self.budget_path = budget_path
        self._key = None
        self.session = requests.Session()

    # -- budget ------------------------------------------------------------------------

    def _state(self) -> dict:
        month = date.today().strftime("%Y-%m")
        if self.budget_path.exists():
            state = json.loads(self.budget_path.read_text())
            if state.get("month") == month:
                return state
        return {"month": month, "calls": 0, "log": []}

    def _record(self, endpoint: str, params: dict) -> None:
        state = self._state()
        state["calls"] += 1
        state["log"].append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "endpoint": endpoint,
            "params": dict(params),
        })
        self.budget_path.parent.mkdir(parents=True, exist_ok=True)
        self.budget_path.write_text(json.dumps(state, indent=2))

    @property
    def calls_used(self) -> int:
        return int(self._state()["calls"])

    @property
    def calls_remaining(self) -> int:
        return self.cfg.api.monthly_call_budget - self.calls_used

    # -- calls -------------------------------------------------------------------------

    def call(self, endpoint: str, cache_key: str, ttl_hours: "float | None" = None,
             **params):
        """Cache-first GET. `cache_key` names the permanent artifact for this request.

        `ttl_hours` is the ONE exception to permanence, and it exists for the in-progress
        season. A completed season is immutable, so caching it forever is correct and is
        what keeps a cold build inside the free tier. The live season is not immutable:
        results land every week, and a cache with no notion of recency would serve the
        first snapshot taken and never update again -- which from the outside looks exactly
        like a working model. Callers pass a TTL only for `cfg.live_season`; see
        `ingest._split_live`.

        None (the default) means permanent, so every historical caller is unaffected.
        """
        path = CACHE_DIR / f"{cache_key}.json"
        if path.exists():
            fresh = ttl_hours is None or (
                time.time() - path.stat().st_mtime < float(ttl_hours) * 3600.0
            )
            if fresh:
                return json.loads(path.read_text())

        if self.calls_remaining <= 0:
            raise APIBudgetExceeded(
                f"Monthly CFBD budget of {self.cfg.api.monthly_call_budget} is exhausted "
                f"({self.calls_used} used). Missing cache key: {cache_key}. Work from "
                "cache until the month rolls over -- this is why the cache is permanent."
            )

        if self._key is None:
            self._key = load_api_key()
        resp = self.session.get(
            f"{BASE_URL}/{endpoint.lstrip('/')}",
            params=params,
            headers={"Authorization": "Bearer " + self._key},
            timeout=120,
        )
        self._record(endpoint, params)
        if resp.status_code == 401:
            raise RuntimeError(
                "CFBD returned 401 -- key missing, expired, or not loaded. Check .env."
            )
        if resp.status_code == 429:
            raise APIBudgetExceeded("CFBD returned 429: rate limited or over quota.")
        resp.raise_for_status()

        data = resp.json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return data

    def frame(self, endpoint: str, cache_key: str, ttl_hours: "float | None" = None,
              **params) -> pd.DataFrame:
        return pd.json_normalize(
            self.call(endpoint, cache_key, ttl_hours=ttl_hours, **params)
        )
