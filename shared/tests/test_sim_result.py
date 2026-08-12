import numpy as np,pytest
from sim_core import SimResult
def lattice():
    h,a=np.meshgrid(np.array([14,17,20,24,27,31]),np.array([10,13,17,20,24,28]),indexing="ij"); h=h.ravel().astype(float); a=a.ravel().astype(float); return SimResult(h-a,h+a,h,a)
def test_calibration_preserves_integer_lattice_and_targets():
    sim=lattice(); calibrated=sim.recentered(1.25).retotaled(44.25)
    assert calibrated.mean_margin==pytest.approx(1.25,abs=1e-6); assert calibrated.mean_total==pytest.approx(44.25,abs=1e-6)
    assert np.array_equal(calibrated.margins,sim.margins) and np.array_equal(calibrated.totals,sim.totals)
def test_push_and_key_number_atoms_survive():
    sim=lattice().recentered(-1.75); pmf=sim.margin_pmf(); assert pmf[3]>0 and pmf[7]>0; assert sim.push_prob(3)==pytest.approx(pmf[3])
def test_weighted_probabilities():
    sim=SimResult(np.array([-3.,3.,7.]),np.array([40.,44.,50.]),np.array([18.,24.,28.]),np.array([21.,21.,21.]),weights=np.array([.1,.2,.7]))
    assert sim.mean_margin==pytest.approx(5.2); assert sim.cover_prob(3,"home")==pytest.approx(.875)
