"""Run the four test suites in isolated interpreters.

Both league repositories intentionally use a top-level package named ``src``. Running
them in one pytest process can bind the second suite to the first league's package, so
this is the supported root test entry point.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(label: str, test_path: str, pythonpath: str) -> int:
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": pythonpath,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    })
    print(f"\n=== {label} ===", flush=True)
    return subprocess.call(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", test_path],
        cwd=ROOT,
        env=env,
    )


def main() -> int:
    suites = [
        ("shared", "shared/tests", str(ROOT / "shared")),
        ("NFL", "nfl-model/tests", os.pathsep.join([str(ROOT / "shared"), str(ROOT / "nfl-model")])),
        ("NCAA", "ncaa-model/tests", os.pathsep.join([str(ROOT / "shared"), str(ROOT / "ncaa-model")])),
        ("Props", "prop-model/tests", os.pathsep.join([str(ROOT / "shared"), str(ROOT / "prop-model")])),
    ]
    return max(run(*suite) for suite in suites)


if __name__ == "__main__":
    raise SystemExit(main())
