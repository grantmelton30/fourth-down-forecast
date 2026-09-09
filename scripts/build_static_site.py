#!/usr/bin/env python3
"""Create a complete, clean Render static artifact in ``dist``."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
DIST = ROOT / "dist"


def require_json(path: Path) -> None:
    """Require a committed publication artifact that parses as a JSON object."""
    if not path.is_file():
        raise RuntimeError(f"publication artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"publication artifact is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"publication artifact must contain a JSON object: {path}")


def main() -> int:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "publish_ledger.py")],
                   check=True)
    explorer = WEB / "api/v1/explorer.json"
    require_json(explorer)
    # track-rules owns this artifact and commits it alongside the append-only logs. Render's
    # static build has no project dependencies installed, so packaging must validate the
    # committed record rather than silently regenerate it with pandas here.
    tracker = WEB / "api/v1/tracker.json"
    require_json(tracker)
    if DIST.resolve().parent != ROOT.resolve() or DIST.name != "dist":
        raise RuntimeError("static output must remain inside this checkout")
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    for name in ("index.html", "styles.css", "app.js", "favicon.svg"):
        shutil.copy2(WEB / name, DIST / name)
    shutil.copytree(WEB / "api", DIST / "api")
    required = [DIST / "index.html", DIST / "styles.css", DIST / "app.js",
                DIST / "favicon.svg",
                DIST / "api/v1/predictions.json", DIST / "api/v1/explorer.json",
                DIST / "api/v1/tracker.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"static build is incomplete: {missing}")
    print(f"static artifact ready: {DIST} ({len(required)} required files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
