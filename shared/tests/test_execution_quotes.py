import json
import pytest
from execution_quotes import validate_quote, matching_quote


def quote():
    return {"league":"ncaa","game_id":"1","book":"Book","source":"manual verified quote",
            "kickoff":"2026-09-10T18:00:00Z","observed_at":"2026-09-10T16:00:00Z",
            "ingested_at":"2026-09-10T16:01:00Z","line":50.5,"over_price":-105,"under_price":-115}


def test_exact_recent_quote_required(tmp_path):
    p=tmp_path/"quotes.jsonl";p.write_text(json.dumps(quote()))
    kw=dict(league="ncaa",game_id="1",kickoff=quote()["kickoff"],book="Book",line=50.5,now="2026-09-10T16:02:00Z")
    assert matching_quote(p,**kw)["under_price"] == -115
    assert matching_quote(p,**{**kw,"book":"Other"}) is None
    assert matching_quote(p,**{**kw,"line":51}) is None
    assert matching_quote(p,**{**kw,"now":"2026-09-10T16:16:00Z"}) is None
    assert matching_quote(p,**{**kw,"now":"2026-09-10T15:59:00Z"}) is None


@pytest.mark.parametrize("change",[{"under_price":0},{"line":float("nan")},
                                   {"ingested_at":"2026-09-10T19:00:00Z"},
                                   {"observed_at":"2026-09-10T16:00:00"}])
def test_invalid_quote_rejected(change):
    with pytest.raises(ValueError): validate_quote({**quote(),**change})
