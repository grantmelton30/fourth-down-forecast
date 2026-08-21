#!/usr/bin/env python3
"""Create a complete, clean Render static artifact in ``dist``."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
DIST = ROOT / "dist"


def main() -> int:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "publish_ledger.py")],
                   check=True)
    explorer = WEB / "api/v1/explorer.json"
    if not explorer.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_explorer.py")],
                       check=True)
    # The forward record is generated the same way, and for the same reason: publish-ledger
    # runs this script on its own 4-hourly schedule without running the trackers, so the
    # artifact has to be able to produce itself rather than silently ship a stale or absent
    # Record tab. It reads the append-only logs, so regenerating is always safe.
    tracker = WEB / "api/v1/tracker.json"
    if not tracker.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_tracker.py")],
                       check=True)
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
