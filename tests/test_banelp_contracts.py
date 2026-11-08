import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiments.banelp.contracts import (DEFAULT_CONFIG, deterministic_split, validate_practical_config,
    validate_runtime_path, validate_tensors)
from experiments.banelp.metrics import project_metrics
from experiments.banelp.data import apply_rir, crop_center, make_sample, prepare_rir
from experiments.banelp.candidates import identity, t1, t2, t5, _stft, _istft
from experiments.banelp.diagnostics import oracle_ratio_mask
from experiments.banelp.diagnostics import oracle_namespace

class ContractTests(unittest.TestCase):
    def test_contract_and_practical_oracle_separation(self):
        validate_practical_config(DEFAULT_CONFIG)
        bad={**DEFAULT_CONFIG,"oracle":True}
        with self.assertRaises(ValueError): validate_practical_config(bad)
        with self.assertRaises(ValueError): validate_tensors(np.zeros((1,8)),np.zeros((4,2,8)))
    def test_split_is_fixed_and_requires_50(self):
        result=deterministic_split([f'track-{i}' for i in range(50)])
        self.assertEqual(list(result.values()).count('dev'),10)
        self.assertEqual(result,deterministic_split([f'track-{i}' for i in reversed(range(50))]))
        with self.assertRaises(ValueError): deterministic_split(['short'])
    def test_runtime_path_is_confined(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(validate_runtime_path(Path(tmp)/'data',tmp).is_relative_to(Path(tmp).resolve()))
            with self.assertRaises(ValueError): validate_runtime_path(Path(tmp).parent/'outside',tmp)
    def test_metrics_known_scale_and_silent(self):
        rng=np.random.default_rng(1); s=rng.normal(size=(2,1024)); others=rng.normal(size=(3,2,1024)); targets=np.concatenate([s[None],others])
        result=project_metrics(s,2*s,targets)
        self.assertAlmostEqual(result['target_gain'],2)
        self.assertEqual(result['si_sdr']['state'],'positive-infinity')
        zero=project_metrics(np.zeros_like(s),s,targets)
        self.assertEqual(zero['state'],'silent-target')
    def test_crop_rms_and_rir_stereo(self):
        dry=np.ones((100,2),np.float32); crop,start=crop_center(dry,60)
        self.assertEqual((start,len(crop)),(20,60))
        wet,info=apply_rir(dry,np.array([1.,.2]))
        with self.assertRaises(ValueError): apply_rir(dry,np.ones((4,4)))
        apply_rir(dry,np.ones((4,4)),rir_channel=0)
        self.assertTrue(np.allclose(wet[:,0],wet[:,1]))
        self.assertAlmostEqual(info['wet_rms_after_gain'],info['dry_rms'],places=5)
        converted,provenance=prepare_rir(np.r_[1.,np.zeros(100)],44100)
        self.assertEqual(provenance['sample_rate'],44100); self.assertEqual(len(converted),101)

class CandidateTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(7); self.est=rng.normal(0,.02,(4,2,8192)).astype(np.float32); self.x=self.est.sum(axis=0)
    def test_identity_t1_t2_t5_shapes_and_finiteness(self):
        self.assertTrue(np.array_equal(identity(self.x,self.est),self.est))
        out=t1(self.x,self.est); self.assertTrue(np.allclose(out.sum(axis=0),self.x,atol=1e-6))
        for out in (t2(self.x,self.est),t5(self.x,self.est)):
            self.assertEqual(out.shape,self.est.shape); self.assertTrue(np.isfinite(out).all())
    def test_stft_roundtrip_channels_and_length(self):
        rng=np.random.default_rng(13); audio=rng.normal(size=(2,8192)).astype(np.float32)
        restored=_istft(_stft(audio),len(audio[0]))
        self.assertEqual(restored.shape,audio.shape)
        self.assertLess(float(np.max(np.abs(restored-audio))),1e-5)
    def test_oracle_requires_targets_and_is_dev_namespaced(self):
        out=oracle_ratio_mask(self.x,self.est,self.est)
        self.assertEqual(out.shape,self.est.shape)
        self.assertEqual(oracle_namespace('dev'),'oracle-dev')
        with self.assertRaises(ValueError): oracle_namespace('evaluation')
