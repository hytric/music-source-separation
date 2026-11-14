import importlib.util,unittest
from pathlib import Path
import numpy as np
from core import project,reconstruction

class ConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parent.parent
        p=next(p for p in (root/'입력전처리검증_2026-11-13/run.py',root/'frontend_validation/run.py') if p.exists())
        s=importlib.util.spec_from_file_location('consistency_test_front',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);cls.front=m
    def test_both_projections_close_sum_and_preserve_protected_bytes(self):
        rng=np.random.default_rng(6);z=rng.normal(0,.03,(4,2,16384)).astype(np.float32);x=rng.normal(0,.04,(2,16384)).astype(np.float32)
        for mode in ('uniform','power'):
            result=project(x,z,mode,self.front.stft,self.front.istft)
            self.assertEqual(result[[0,3]].tobytes(),z[[0,3]].tobytes())
            self.assertLess(reconstruction(x,result)['max_abs'],1e-6)
    def test_consistent_input_unchanged_within_roundtrip_error(self):
        z=np.random.default_rng(9).normal(0,.02,(4,2,16384)).astype(np.float32);x=z.sum(0,dtype=np.float64).astype(np.float32)
        for mode in ('uniform','power'):
            result=project(x,z,mode,self.front.stft,self.front.istft)
            np.testing.assert_allclose(result,z,atol=1e-6,rtol=1e-6)
    def test_silent_free_sources_get_finite_equal_residual(self):
        z=np.zeros((4,2,16384),np.float32);z[0]=.1;x=np.full((2,16384),.14,np.float32)
        for mode in ('uniform','power'):
            result=project(x,z,mode,self.front.stft,self.front.istft)
            self.assertTrue(np.isfinite(result).all());np.testing.assert_allclose(result[1],.02,atol=1e-6);np.testing.assert_allclose(result[2],.02,atol=1e-6)
    def test_bad_contract_rejected(self):
        z=np.zeros((4,2,16384),np.float32);x=np.zeros((2,16384),np.float32)
        with self.assertRaises(ValueError):project(x,z[:3],'uniform')
        z[1,0,0]=np.nan
        with self.assertRaises(ValueError):project(x,z,'uniform')

class SafetyTests(unittest.TestCase):
    def test_median_zero_does_not_hide_one_bad_dry_song(self):
        import json
        from report import summarize
        config=json.loads(Path(__file__).with_name('protocol.json').read_text());rows=[]
        for condition in config['conditions']:
            for i in range(10):
                def score(values):return {'metrics':[{'stem':stem,'state':'ok','si_sdr':value,'si_sar':0.,'target_gain':.5} for stem,value in zip(config['sources'],values)]}
                baseline=score([0.]*4)
                candidate=score([0.,2.,2.,0.] if condition=='wet_R4' else ([0.,-4.,-4.,0.] if i==0 else [0.]*4))
                rows.append({'track':str(i),'condition':condition,
                    'scores':{name:(candidate if name in ('selected_power','selected_uniform') else baseline) for name in config['methods']},
                    'checks':{name:{'protected_max_abs':0.,'full40':{'max_abs':0.},'core30':{'relative_residual_rms':0.}} for name in config['methods']},
                    'baseline_metric_max_error':0.,'selected_metric_max_error':0.,'legacy_metric_max_error':0.})
        summary=summarize(rows,config)
        self.assertEqual(summary['comparisons']['dry']['selected_power']['baseline']['overall']['median'],0.)
        self.assertTrue(summary['gates']['wet_gain_vs_baseline'])
        self.assertFalse(summary['gates']['each_dry_song']);self.assertFalse(summary['accepted'])

if __name__=='__main__':unittest.main()
