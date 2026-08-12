import pandas as pd
from cache_manifest import build_signature,frame_signature,read_parquet,write_parquet
def test_manifest_invalidates_changed_inputs(tmp_path):
    builder=tmp_path/"builder.py"; builder.write_text("x=1\n"); (tmp_path/"src").mkdir()
    frame=pd.DataFrame({"id":[1,2],"v":[3,4]}); sig=build_signature(builder=builder,config={"x":1},inputs=frame_signature(frame))
    path=tmp_path/"x.parquet"; write_parquet(frame,path,sig); assert read_parquet(path,["id"],sig) is not None
    changed=build_signature(builder=builder,config={"x":2},inputs=frame_signature(frame)); assert read_parquet(path,["id"],changed) is None
