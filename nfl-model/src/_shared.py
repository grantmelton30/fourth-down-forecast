"""Locate the shared, sport-agnostic simulator.

`../shared/sim_core.py` holds the one drive-level Monte Carlo used by both this repo and
ncaa-model. It cannot be imported as a normal package because both repos have a top-level
`src` package and the names would collide, so the sibling directory goes on sys.path here
and nowhere else -- every module that needs the simulator does `from ._shared import
sim_core`.
"""

from __future__ import annotations

import sys
from pathlib import Path

SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

import robust_inference  # noqa: E402
import sim_core  # noqa: E402
import validated_model  # noqa: E402

__all__ = ["robust_inference", "sim_core", "validated_model", "SHARED_DIR"]
