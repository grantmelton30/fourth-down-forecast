from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_scheduled_refresh_rebuilds_models_and_publishes_only_after_validation():
    workflow = (ROOT / ".github/workflows/publish.yml").read_text()
    assert "actions/cache" in workflow
    assert "if: always()" in workflow
    assert "CFBD_API_KEY: ${{ secrets.CFBD_API_KEY }}" in workflow
    assert "scripts/refresh_publication.py" in workflow
    assert "scripts/validate_publication.py" in workflow
    assert "concurrency:" in workflow
    assert "scripts/publish_ledger.py" not in workflow


def test_refresh_workflow_keeps_manual_and_twice_weekly_triggers():
    workflow = (ROOT / ".github/workflows/publish.yml").read_text()
    assert "workflow_dispatch:" in workflow
    assert 'cron: "17 11 * * 2,5"' in workflow
    assert 'cron: "17 13 * * *"' in workflow


def test_ui_reserves_warning_icon_for_extreme_disagreement_and_labels_missing_market():
    app = (ROOT / "web/app.js").read_text()
    assert "r.out_of_distribution?'" in app
    assert "No line posted yet" in app
    assert "r.warnings?.length?'" not in app
