from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_scheduled_refresh_rebuilds_models_and_publishes_only_after_validation():
    workflow = (ROOT / ".github/workflows/publish.yml").read_text()
    assert "actions/cache" in workflow
    assert "if: always()" in workflow
    assert "CFBD_API_KEY: ${{ secrets.CFBD_API_KEY }}" in workflow
    assert "scripts/refresh_publication.py" in workflow
    assert "scripts/refresh_market_data.py" in (
        ROOT / "scripts/refresh_publication.py"
    ).read_text()
    assert "scripts/validate_publication.py" in workflow
    assert "concurrency:" in workflow
    assert "scripts/publish_ledger.py" not in workflow
    assert 'pytest -q -m "not integration" tests' in workflow


def test_refresh_workflow_keeps_manual_and_twice_weekly_triggers():
    workflow = (ROOT / ".github/workflows/publish.yml").read_text()
    assert "workflow_dispatch:" in workflow
    assert 'cron: "17 11 * * 2,5"' in workflow
    # Market-only republish, bumped from once daily to every 4 hours 2026-08-18 -- no
    # model rebuild, just fresh live-season lines, cheap enough to run this often. The
    # schedule string and the step's own comparison must stay in sync or the market-only
    # branch silently stops matching and every scheduled run does a full rebuild instead.
    assert 'cron: "17 */4 * * *"' in workflow
    assert 'github.event.schedule }}" = "17 */4 * * *"' in workflow


def test_ui_reserves_warning_icon_for_extreme_disagreement_and_labels_missing_market():
    app = (ROOT / "web/app.js").read_text()
    assert "r.out_of_distribution?'" in app
    assert "No line posted yet" in app
    assert "r.warnings?.length?'" not in app
