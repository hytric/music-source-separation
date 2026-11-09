import unittest
import numpy as np
from experiments.banelp.candidates import (DONOR_EVIDENCE_STATUS, _istft, _stft, candidate_registry, consistency_iteration, donor_reseparation, mixture_consistency,
                                           parse_model_candidate, run_candidate)
from experiments.banelp.selection import holm_adjust, select_candidate, training_cost_record

class CandidatePhysicsTests(unittest.TestCase):
    def test_donor_reseparation_moves_bass_from_the_donor_and_preserves_the_sum(self):
        rng=np.random.default_rng(9); estimates=rng.normal(size=(4,2,2048)).astype(np.float32); x=estimates.sum(axis=0)
        calls=[]
        def separator(audio):
            calls.append(np.array(audio)); a=np.asarray(audio)
            return np.stack([a*1,a*2,a*3,a*4]).astype(np.float32)  # distinct per-stem scales: wrong source index/strength cannot cancel out
        for donors,idx in ((("other",),3),(("drums",),1)):
            for strength in (0.0,.5,1.0):
                calls.clear(); out=donor_reseparation(x,estimates,separator,donors=donors,strength=strength)
                delta=strength*3*estimates[idx]
                self.assertTrue(np.allclose(out[2],estimates[2]+delta,atol=1e-6)); self.assertTrue(np.allclose(out[idx],estimates[idx]-delta,atol=1e-6))
                for k in range(4):
                    if k not in (2,idx): self.assertTrue(np.array_equal(out[k],estimates[k]))
                self.assertTrue(np.allclose(out.sum(axis=0),estimates.sum(axis=0),atol=1e-5,rtol=0))
                self.assertEqual(len(calls),1); self.assertTrue(np.array_equal(calls[0],estimates[idx]))  # the donor stem estimate is what is re-separated
        both=donor_reseparation(x,estimates,separator,donors=("drums","other"),strength=.5)
        self.assertTrue(np.allclose(both[2],estimates[2]+.5*3*(estimates[1]+estimates[3]),atol=1e-5)); self.assertTrue(np.allclose(both.sum(axis=0),estimates.sum(axis=0),atol=1e-5,rtol=0))
        self.assertTrue(np.array_equal(donor_reseparation(x,estimates,separator,donors=("other",),strength=0.0),estimates))
    def test_donor_reseparation_rejects_bad_arguments_and_bad_separator_output(self):
        rng=np.random.default_rng(9); estimates=rng.normal(size=(4,2,64)).astype(np.float32); x=estimates.sum(axis=0)
        good=lambda a:np.stack([a]*4)
        for kwargs in ({"donors":("vocals",)},{"donors":()},{"strength":1.5},{"strength":-.1}):
            with self.assertRaises(ValueError): donor_reseparation(x,estimates,good,**{"donors":("other",),"strength":.5,**kwargs})
        with self.assertRaises(ValueError): donor_reseparation(x,estimates,lambda a:np.stack([a]*3),donors=("other",))
        with self.assertRaises(ValueError): donor_reseparation(x,estimates,lambda a:np.full((4,2,64),np.nan,np.float32),donors=("other",))
    def test_donor_candidates_are_marked_exploratory_and_depend_on_the_baseline(self):
        registry=candidate_registry(baseline_id="htdemucs")
        for name in ("C-donor-other-lambda05","C-donor-drums-lambda05"):
            self.assertEqual(registry[name]["params"]["evidence_status"],DONOR_EVIDENCE_STATUS); self.assertEqual(registry[name]["models"],["htdemucs"])
        self.assertIn("no direct precedent for 4-stem frozen music separators",DONOR_EVIDENCE_STATUS); self.assertEqual(registry["T1-uniform"]["models"],[])
    def test_model_candidates_replace_and_average(self):
        rng=np.random.default_rng(2); x=rng.normal(size=(2,128)).astype(np.float32)
        base=rng.normal(size=(4,2,128)).astype(np.float32); a=rng.normal(size=(4,2,128)).astype(np.float32); b=rng.normal(size=(4,2,128)).astype(np.float32)
        registry=candidate_registry(model_candidates=["replace:a","average:a+b","average:a+b+base"],baseline_id="base"); by={"base":base,"a":a,"b":b}
        self.assertTrue(np.array_equal(run_candidate(registry["replace:a"],x,base,by),a))
        self.assertTrue(np.allclose(run_candidate(registry["average:a+b"],x,base,by),(a+b)/2,atol=1e-6))
        self.assertTrue(np.allclose(run_candidate(registry["average:a+b+base"],x,base,by),(a+b+base)/3,atol=1e-6))
        self.assertEqual((registry["replace:a"]["models"],registry["average:a+b+base"]["models"]),(["a"],["a","b","base"]))
        self.assertEqual(registry["average:a+b"]["params"],{"kind":"average","models":["a","b"],"weights":"equal"})
        with self.assertRaisesRegex(ValueError,"missing"): run_candidate(registry["average:a+b"],x,base,{"a":a})
        self.assertTrue(np.array_equal(run_candidate(registry["T1-uniform"],x,base),mixture_consistency(x,base)))  # ordinary candidates ignore by_model
    def test_model_candidate_name_rules(self):
        self.assertEqual(parse_model_candidate("average:a+b"),("average",["a","b"])); self.assertEqual(parse_model_candidate("replace:a"),("replace",["a"]))
        for bad in ("replace:a+b","average:a","average:b+a","average:a+a","blend:a","replace:","a","average:a++b"):
            with self.assertRaises(ValueError,msg=bad): parse_model_candidate(bad)
        with self.assertRaisesRegex(ValueError,"identity"): candidate_registry(model_candidates=["replace:base"],baseline_id="base")
        with self.assertRaises(ValueError): candidate_registry(model_candidates=["replace:a","replace:a"],baseline_id="base")
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
