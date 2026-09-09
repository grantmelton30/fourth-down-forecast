#!/usr/bin/env python3
"""Validate the generated tracking API after it has been written to disk."""
from __future__ import annotations

import json

from build_tracker import OUT, validate_payload


def main() -> int:
    try:
        payload = json.loads(OUT.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read generated tracker: {OUT}") from exc
    validate_payload(payload)
    print(f"validated tracker schema 2: {len(payload['bets'])} unique source rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
