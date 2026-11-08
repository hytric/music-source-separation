import unittest
import numpy as np
from experiments.banelp.candidates import _istft, _stft, consistency_iteration, mixture_consistency
from experiments.banelp.selection import holm_adjust, select_candidate, training_cost_record

class CandidatePhysicsTests(unittest.TestCase):
    def test_donor_redistribution_preserves_estimate_sum(self):
        rng=np.random.default_rng(9); estimates=rng.normal(size=(4,2,2048)).astype(np.float32)
        donor=estimates[3].copy(); bass=estimates[2].copy(); delta=.5*donor
        changed=estimates.copy(); changed[2]=bass+delta; changed[3]=donor-delta
        self.assertTrue(np.allclose(changed.sum(axis=0),estimates.sum(axis=0),atol=1e-6,rtol=0))
    def test_holm_and_identity_fallback_and_unknown_training_cost(self):
        self.assertEqual(holm_adjust({'a':.01,'b':.04}),{'a':.02,'b':.04})
        choice=select_candidate([],sar_policy_status='confirmed',sar_floor_db=0)
        self.assertEqual(choice['selected'],'identity')
        self.assertEqual(training_cost_record()['status'],'unmeasured')
        with self.assertRaises(ValueError): select_candidate([],sar_policy_status='pending-owner-confirmation',sar_floor_db=None)
    def test_consistency_zero_is_identity_and_nonzero_finite(self):
        rng=np.random.default_rng(10); s=rng.normal(size=(4,2,8192)).astype(np.float32); x=s.sum(axis=0)
        self.assertTrue(np.array_equal(consistency_iteration(x,s,0),s))
        self.assertTrue(np.isfinite(consistency_iteration(x,s,1)).all())
    def test_istft_accepts_channel_and_source_stacks(self):
        rng=np.random.default_rng(11); audio=rng.normal(size=(4,2,6000)).astype(np.float32)
        self.assertLess(np.max(np.abs(_istft(_stft(audio[0]),6000)-audio[0])),1e-5)
        stacked=_istft(np.stack([_stft(a) for a in audio]),6000)
        self.assertEqual(stacked.shape,audio.shape); self.assertLess(np.max(np.abs(stacked-audio)),1e-5)
        with self.assertRaises(ValueError): _istft(_stft(audio[0])[0],6000)
