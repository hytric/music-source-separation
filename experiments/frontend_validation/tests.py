import sys
import unittest
import numpy as np

sys.path.insert(0,'/home/iiplab/code/music-source-separation/.agent_runtime/frontend-verify-v1/vendor')
from run import stft,istft,restore,wpe_front,paired_stat


class FrontendContracts(unittest.TestCase):
    def test_stft_alignment_and_length(self):
        x=np.random.default_rng(1).normal(size=(2,16385)).astype(np.float32)*.1
        y=istft(stft(x),x.shape[-1]);self.assertEqual(y.shape,x.shape)
        np.testing.assert_allclose(y,x,atol=2e-7,rtol=1e-5)

    def test_residual_allocation_keeps_original_mix(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(2,32768)).astype(np.float32)*.1
        z=rng.normal(size=(4,2,32768)).astype(np.float32)*.01
        for weighted in [False,True]:
            y=restore(x,z,weighted);np.testing.assert_allclose(y.sum(axis=0),x,atol=1e-6,rtol=1e-5)

    def test_silent_restoration(self):
        x=np.zeros((2,16384),np.float32);z=np.zeros((4,2,16384),np.float32)
        self.assertEqual(float(np.abs(restore(x,z,True)).max()),0.)

    def test_wpe_silent_is_finite(self):
        x=np.zeros((2,65536),np.float32);y=wpe_front(x)
        self.assertTrue(np.isfinite(y).all());self.assertEqual(y.shape,x.shape)
        self.assertEqual(float(np.abs(y).max()),0.)

    def test_paired_bootstrap_has_no_fake_samples(self):
        x=paired_stat([1.,1.,1.,None]);self.assertEqual(x['n'],3)
        self.assertEqual(x['excluded'],1);self.assertEqual(x['ci'],[1.,1.])


if __name__=='__main__':unittest.main()
