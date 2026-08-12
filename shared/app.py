"""The combined viewer: both sports, one page, one process.

    streamlit run shared/app.py

This is the entrypoint VIEWER_SPEC.md §3 calls for and which was never built. The two
per-repo `app.py` shims still work and still show a single sport; they are unchanged.

WHY THIS FILE IS NOT JUST A RADIO BUTTON. nfl-model and ncaa-model each have a top-level
package called `src`. Python caches by name, so once `src.config` has been imported from
one repo, `import src.config` from the other silently returns the FIRST repo's module --
and the viewer would cheerfully label NFL numbers as college ones. `_activate()` below is
the whole reason this file exists: on every switch it drops every `src*` module and
re-points `sys.path` at the active repo, so the import machinery cannot cross the streams.

Streamlit re-runs this script top to bottom on every interaction, which is what makes that
safe: there is exactly one active sport per run.

MOBILE. The controls used to live entirely in `st.sidebar`, which a phone collapses behind
a hamburger -- so the page opened on a metric card reading "Ohio State" with no visible way
to change anything. The sport switch is in the main body, and the sidebar starts expanded.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

SHARED = Path(__file__).resolve().parent
ROOT = SHARED.parent

SPORTS = {
    "NFL": {"repo": ROOT / "nfl-model", "env": "NFL_MODEL_CACHE_DIR",
            "cache": Path.home() / ".cache" / "nfl-model"},
    "NCAA": {"repo": ROOT / "ncaa-model", "env": "NCAA_MODEL_CACHE_DIR",
             "cache": Path.home() / ".cache" / "ncaa-model"},
}

# Both cache dirs must be set BEFORE either repo's config module is imported -- each reads
# its own at import time, and a wrong default points the viewer at the repo-local
# data/cache, which on this machine is a stale partial copy.
for _meta in SPORTS.values():
    if not os.environ.get(_meta["env"]) and _meta["cache"].exists():
        os.environ[_meta["env"]] = str(_meta["cache"])

# Must be the first Streamlit call on the page. view.py skips its own when embedded.
st.set_page_config(
    page_title="Football model viewer",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _activate(sport: str, purge: bool = True) -> None:
    """Make `sport`'s repo the one `import src...` resolves to.

    Drops cached `src*` modules and removes the other repo from sys.path. Without this the
    second sport selected in a session would render the first sport's data.
    """
    repo = str(SPORTS[sport]["repo"])
    if purge:
        for name in [m for m in list(sys.modules) if m == "src" or m.startswith("src.")]:
            del sys.modules[name]
    for meta in SPORTS.values():
        p = str(meta["repo"])
        while p in sys.path:
            sys.path.remove(p)
    sys.path.insert(0, repo)
    if str(SHARED) not in sys.path:
        sys.path.insert(0, str(SHARED))
    os.environ["SPORT_MODEL"] = sport.lower()


def main() -> None:
    st.markdown(
        """<div class="product-header">
        <div class="product-kicker">GRIDIRON INTELLIGENCE</div>
        <div class="product-title">Game Forecast Lab</div>
        <div class="product-subtitle">Causal team ratings, score distributions and market diagnostics</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # In the main body, not the sidebar: on a phone the sidebar is hidden behind a
    # hamburger and the page looks like it has no controls at all.
    labels = list(SPORTS)
    if hasattr(st, "segmented_control"):
        # Label hidden: the two buttons read as NFL / NCAA on their own, and a "Sport"
        # caption above them was one more piece of furniture between the title and the page.
        choice = st.segmented_control(
            "Sport", labels, default=labels[0], key="sport_choice",
            label_visibility="collapsed",
        ) or labels[0]
    else:
        choice = st.radio(
            "Sport", labels, horizontal=True, label_visibility="collapsed",
            key="sport_choice",
        )

    missing = [s for s, m in SPORTS.items() if not m["repo"].exists()]
    if missing:
        st.warning(f"Repo not found for: {', '.join(missing)}")
    if not SPORTS[choice]["repo"].exists():
        st.error(f"{choice}: {SPORTS[choice]['repo']} does not exist.")
        return

    # Only tear down module state when the sport actually changes. view.get_adapter is
    # behind st.cache_resource and the NCAA adapter holds a drive model fit on ~120k
    # drives; re-importing on every interaction would throw that away and refit each click.
    changed = st.session_state.get("_active_sport") != choice
    _activate(choice, purge=changed)
    if changed:
        for name in ("view", "sport"):
            sys.modules.pop(name, None)
        st.session_state["_active_sport"] = choice
    os.environ["VIEWER_EMBEDDED"] = "1"

    # Imported after _activate so `view` and its adapter resolve the right `src`.
    import view

    view.main()


main()
