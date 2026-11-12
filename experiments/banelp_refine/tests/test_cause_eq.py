import unittest
import numpy as np

from cause_eq.dsp import factors,apply_filter,known_q,blind_q,log_power,CORE
from cause_eq.metrics import score_sets,equivalence_check


class CauseContracts(unittest.TestCase):
    def test_magnitude_phase_factorization(self):
        h=np.zeros(100);h[7]=1;h[23]=.4;h[76]=-.1
        responses,magnitude,check=factors(h,7,1024)
        self.assertLess(check['relative_reconstruction'],1e-10)
        self.assertLess(check['unit_magnitude'],1e-10)
        np.testing.assert_allclose(np.abs(responses['M']),np.abs(responses['H']),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(np.abs(responses['A']),1,atol=1e-12)

    def test_filter_roundtrip_same_grid(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(2,8192)).astype(np.float32)
        q=np.exp(.3*np.cos(np.linspace(0,np.pi,4097)))
        y=apply_filter(apply_filter(x,q),1/q)
        np.testing.assert_allclose(x,y,atol=5e-7,rtol=1e-5)

    def test_extended_fft_keeps_position_and_reports_energy(self):
        x=np.zeros((2,2048),dtype=np.float32);x[:,1000]=1
        q=np.ones(4097)*1.5;stats={}
        y=apply_filter(x,q,stats)
        np.testing.assert_allclose(y,1.5*x,atol=1e-7)
        self.assertLess(stats['maximum_discarded_energy_fraction'],1e-12)

    def test_gain_is_bounded_and_outside_band_is_identity(self):
        h=np.zeros(50);h[3]=1;h[12]=.8
        _,m,_=factors(h,3,8192);q=known_q(m,8192)
        f=np.fft.rfftfreq(8192,1/44100)
        self.assertGreaterEqual(q.min(),10**(-6/20)-1e-12)
        self.assertLessEqual(q.max(),10**(6/20)+1e-12)
        np.testing.assert_array_equal(q[(f<=10)|(f>=1500)],1)

    def test_blind_eq_uses_only_input_and_frozen_reference(self):
        x=np.random.default_rng(2).normal(size=(2,CORE)).astype(np.float32)
        _,reference=log_power(x)
        q=blind_q(x,reference,8192)
        np.testing.assert_allclose(q,1,atol=1e-12)

    def test_fast_metrics_match_legacy_and_silent_stem(self):
        y=np.random.default_rng(4).normal(size=(4,2,8192));y[0]=0
        z=y+.1*np.random.default_rng(5).normal(size=y.shape)
        self.assertLess(equivalence_check(y,z),1e-8)
        self.assertEqual(score_sets(y,{'x':z})['x']['metrics'][0]['state'],'silent-target')


if __name__=='__main__':unittest.main()
