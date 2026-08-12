import numpy as np,pytest
from robust_inference import fit_ols,season_block_bootstrap
def test_clustered_se_reflects_week_correlation():
    rng=np.random.default_rng(7); weeks=np.repeat(np.arange(40),12); xw=rng.normal(size=40); x=np.repeat(xw,12)+rng.normal(scale=.05,size=len(weeks)); y=.5+.4*x+np.repeat(rng.normal(scale=2,size=40),12)+rng.normal(scale=.3,size=len(weeks))
    iid=fit_ols(y,x,covariance="iid"); clustered=fit_ols(y,x,covariance="cluster",groups=weeks)
    assert clustered.n_clusters==40 and clustered.se[1]>iid.se[1]
def test_free_intercept():
    x=np.tile([-2.,-1.,1.,2.],25); fit=fit_ols(3.25+.6*x,x)
    assert fit.beta[0]==pytest.approx(3.25) and fit.beta[1]==pytest.approx(.6)
def test_block_bootstrap_deterministic():
    values=np.array([0.,0.,1.,1.,4.,4.]); seasons=[1,1,2,2,3,3]
    a=season_block_bootstrap(values,seasons,n_boot=100,seed=4); b=season_block_bootstrap(values,seasons,n_boot=100,seed=4)
    assert np.array_equal(a.replicates,b.replicates)
