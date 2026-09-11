import pandas as pd
from cache_manifest import build_signature,frame_signature,read_parquet,write_parquet
def test_manifest_invalidates_changed_inputs(tmp_path):
    builder=tmp_path/"builder.py"; builder.write_text("x=1\n"); (tmp_path/"src").mkdir()
    frame=pd.DataFrame({"id":[1,2],"v":[3,4]}); sig=build_signature(builder=builder,config={"x":1},inputs=frame_signature(frame))
    path=tmp_path/"x.parquet"; write_parquet(frame,path,sig); assert read_parquet(path,["id"],sig) is not None
    changed=build_signature(builder=builder,config={"x":2},inputs=frame_signature(frame)); assert read_parquet(path,["id"],changed) is None

def test_ncaa_backtest_frame_fails_closed_on_a_foreign_config(tmp_path, monkeypatch):
    """A frame this config did not build must read as absent, not as evidence.

    `walk_forward` already refuses a mismatched manifest, but that guard lives in the
    BUILDER; the adapter used to `pd.read_parquet` straight past it, so the college
    accuracy precondition inside `bets_allowed()` could grade a retired model.
    """
    import sys
    from dataclasses import asdict, dataclass
    from pathlib import Path

    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import cache_manifest
    from sport import NCAAAdapter

    @dataclass
    class Cfg:
        knob: int

    path = tmp_path / "backtest_frame_default.parquet"
    signature = cache_manifest.build_signature(
        builder=Path(cache_manifest.__file__), config=asdict(Cfg(1)), inputs={})
    cache_manifest.write_parquet(
        pd.DataFrame({"game_id": ["1"], "model_spread": [1.0]}), path, signature)

    adapter = NCAAAdapter.__new__(NCAAAdapter)
    monkeypatch.setattr(NCAAAdapter, "_cache_dir", lambda self: tmp_path)
    monkeypatch.setattr(NCAAAdapter, "_cfg", lambda self: self._cache["cfg"])

    # The config that produced the frame still reads it.
    adapter._cache = {"cfg": Cfg(1)}
    assert adapter.backtest_frame() is not None

    # A different configuration gets nothing, rather than a frame it did not build.
    adapter._cache = {"cfg": Cfg(2)}
    assert adapter.backtest_frame() is None

    # An unmanifested frame cannot vouch for itself either.
    adapter._cache = {"cfg": Cfg(1)}
    cache_manifest.manifest_path(path).unlink()
    assert adapter.backtest_frame() is None

    # Nor can a corrupt manifest.
    cache_manifest.write_parquet(
        pd.DataFrame({"game_id": ["1"], "model_spread": [1.0]}), path, signature)
    cache_manifest.manifest_path(path).write_text("{not json")
    assert adapter.backtest_frame() is None
