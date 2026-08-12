"""NCAA launcher for the shared viewer.

    streamlit run app.py

Identical to the NFL launcher apart from the sport argument -- the viewer in
../shared/view.py is sport-agnostic.

Per the amended build order (ingest -> ratings -> viewer -> simulator -> backtest -> gates
-> bet sheet), this is expected to run against a partially built repo. The Ratings tab
lights up as soon as ratings.py produces a walk-forward artifact; the Game explorer stays
dark until the simulator exists. Neither state is an error.

The bet sheet is gated behind GATE_BLEND_INFORMATIVE here exactly as it is for NFL, and
the kill criterion in NCAA_PLAYBOOK.md §9 applies: if held-out t(b_model) fails to clear
2.0 on totals, on the restricted universe, against opening lines, the build stops.
"""

import os
import runpy
import sys
from pathlib import Path

SHARED = Path(__file__).resolve().parent.parent / "shared"
sys.path.insert(0, str(SHARED))
os.environ.setdefault("SPORT_MODEL", "ncaa")

runpy.run_path(str(SHARED / "view.py"), run_name="__main__")
