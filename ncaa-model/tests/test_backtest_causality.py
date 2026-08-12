import numpy as np,pandas as pd
from src.backtest import project_walkforward
def features():
    i=np.arange(360); train=pd.DataFrame({"game_id":[f"t{x}" for x in i],"season":2019,"week":4+i%10,"net_diff":((i%25)-12)/3,"is_home":(i%4!=0).astype(float),"pace_sum":21+(i%13)/3,"eff_sum":(((i*7)%19)-9)/20})
    train["actual_margin"]=2+1.7*train.net_diff+2.2*train.is_home+np.sin(i); train["actual_total"]=31+.8*train.pace_sum+4*train.eff_sum+np.cos(i); train["spread_open"]=3*train.actual_margin; train["total_open"]=3*train.actual_total
    test=pd.DataFrame({"game_id":["past-a","past-b","future-a","future-b"],"season":2020,"week":[4,5,6,7],"net_diff":[-1.,1.,40.,60.],"is_home":[1.,0.,1.,0.],"pace_sum":[22.,24.,70.,90.],"eff_sum":[-.2,.2,5.,8.],"actual_margin":np.nan,"actual_total":np.nan,"spread_open":np.nan,"total_open":np.nan})
    return pd.concat([train,test],ignore_index=True)
def past(frame):
    out=project_walkforward(frame,cfg=None); cols=["game_id","model_spread","model_total","model_spread_market_adjusted","model_total_market_adjusted"]
    return out[out.game_id.isin(["past-a","past-b"])][cols].sort_values("game_id").reset_index(drop=True)
def test_future_rows_cannot_change_published_projection():
    f=features(); full=project_walkforward(f,cfg=None); target=full[full.season==2020]
    np.testing.assert_allclose(target.model_spread,target.model_spread_pure); np.testing.assert_allclose(target.model_total,target.model_total_pure)
    truncated=f[(f.season<2020)|(f.week<=5)]; changed=f.copy(); mask=(changed.season==2020)&(changed.week>5); changed.loc[mask,["net_diff","pace_sum","eff_sum"]]*=-11
    pd.testing.assert_frame_equal(past(truncated),past(f)); pd.testing.assert_frame_equal(past(changed),past(f))
