"""NFL launcher for the shared viewer.

    streamlit run app.py

The viewer itself lives in ../shared/view.py and is sport-agnostic; this file only pins
the sport argument. The NCAA repo gets an identical two-line launcher.

Picks are not shown for NFL: GATE_BLEND_INFORMATIVE fails on this model by design (see
APPENDIX A of NFL_PLAYBOOK.md), and the gate is enforced in shared/sport.py where no UI
path can route around it.
"""

import os
import runpy
import sys
from pathlib import Path

SHARED = Path(__file__).resolve().parent.parent / "shared"
sys.path.insert(0, str(SHARED))
os.environ.setdefault("SPORT_MODEL", "nfl")

runpy.run_path(str(SHARED / "view.py"), run_name="__main__")
