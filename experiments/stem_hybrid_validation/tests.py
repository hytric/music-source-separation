import unittest
import numpy as np
from core import combine,reconstruction,mean_si

class HybridTests(unittest.TestCase):
    def test_stem_order_exact_copy_and_no_alias(self):
        b=np.arange(4*2*31,dtype=np.float32).reshape(4,2,31);f=-b-7
        h=combine(b,f)
        np.testing.assert_array_equal(h[[0,3]],b[[0,3]])
        np.testing.assert_array_equal(h[1:3],f[1:3])
        h[0]=99;self.assertFalse(np.array_equal(h[0],b[0]))
        np.testing.assert_array_equal(f,-b-7)
    def test_invalid_shape_and_nonfinite_rejected(self):
        b=np.zeros((4,2,31),np.float32)
        with self.assertRaises(ValueError):combine(b,b[...,:30])
        f=b.copy();f[1,0,0]=np.nan
        with self.assertRaises(ValueError):combine(b,f)
    def test_reconstruction_flags_cross_model_double_count(self):
        b=np.zeros((4,2,31),np.float32);b[0]=1;b[1]=1;x=b.sum(0)
        f=b.copy();f[1]=2
        self.assertEqual(reconstruction(x,b)['relative_residual_rms'],0.)
        self.assertAlmostEqual(reconstruction(x,combine(b,f))['relative_residual_rms'],.5)
    def test_oracle_score_excludes_silent_stem(self):
        self.assertEqual(mean_si({'metrics':[{'si_sdr':2.},{'si_sdr':None},{'si_sdr':4.},{'si_sdr':6.}]}),4.)

class AcceptanceTests(unittest.TestCase):
    def test_dry_damage_fails_even_when_oracle_passes(self):
        import json
        from pathlib import Path
        from report import summarize
        config=json.loads(Path(__file__).with_name('protocol.json').read_text())
        rows=[]
        for condition in ('dry','wet_R4'):
            for i in range(10):
                def score(values):return {'metrics':[{'stem':s,'si_sdr':v,'si_sar':0.,'target_gain':.5} for s,v in zip(config['sources'],values)]}
                b=score([0.]*4);h=score([0.,2.,2.,0.] if condition=='wet_R4' else [0.,-2.,-2.,0.])
                choice='baseline' if condition=='dry' else 'hybrid';diag=b if condition=='dry' else h
                rows.append({'condition':condition,'scores':{'baseline':b,'finetuned':h,'hybrid':h,'condition_label_route':diag,'gt_case_oracle':diag},
                    'reconstruction':{name:{'relative_residual_rms':0.} for name in config['methods']},
                    'diagnostic_choices':{'condition_label_route':choice,'gt_case_oracle':choice},
                    'checks':{key:0. for key in ('protected_stem_max_abs','baseline_metric_max_error','finetuned_metric_max_error','spliced_metric_max_error','legacy_metric_max_error')}})
        result=summarize(rows,config)
        self.assertTrue(result['gates']['wet_primary']);self.assertFalse(result['gates']['dry_overall'])
        self.assertFalse(result['accepted']);self.assertEqual(result['comparisons']['dry']['gt_case_oracle']['overall']['median'],0.)

if __name__=='__main__':unittest.main()
