import unittest
import torch
from core import normalized_pair,reconstruction_loss

class TrainingMathTests(unittest.TestCase):
    config={'n_fft':64,'hop_length':16,'win_length':64,'normalized':True}
    def test_normalization_inverse_and_source_order(self):
        torch.manual_seed(3);y=torch.randn(1,4,2,256)*.03+.005
        x,t=normalized_pair(y)
        mono=y.sum(1).mean(1,keepdim=True);mean=mono.mean(-1,keepdim=True);std=mono.std(-1,keepdim=True)
        torch.testing.assert_close(x*std+mean,y.sum(1))
        torch.testing.assert_close(t*std[:,None]+mean[:,None],y)
    def test_exact_reconstruction_has_finite_gradients(self):
        t=torch.randn(1,4,2,256);a=t.clone().requires_grad_()
        loss,_=reconstruction_loss(a,t,self.config,guard=16);loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all());self.assertEqual(a.grad.abs().max().item(),0.)
    def test_guards_are_excluded_and_vocals_have_double_weight(self):
        t=torch.zeros(1,4,2,256);a=t.clone();a[...,:16]=10;a[...,-16:]=10
        base,_=reconstruction_loss(t,t,self.config,guard=16)
        guard,_=reconstruction_loss(a,t,self.config,guard=16);torch.testing.assert_close(guard,base)
        a=t.clone();a[:,0,:,64:128]=.1;v,_=reconstruction_loss(a,t,self.config,guard=16)
        a=t.clone();a[:,2,:,64:128]=.1;b,_=reconstruction_loss(a,t,self.config,guard=16)
        torch.testing.assert_close(v-base,2*(b-base))
    def test_silence_fails_explicitly(self):
        with self.assertRaises(ValueError):normalized_pair(torch.zeros(1,4,2,256))

class RetentionGateTests(unittest.TestCase):
    def test_large_total_gain_cannot_hide_vocal_damage(self):
        import json
        import numpy as np
        from pathlib import Path
        from report import summarize
        config=json.loads(Path(__file__).with_name('protocol.json').read_text())
        rows=[]
        for condition in ('dry','wet_R4'):
            for i in range(10):
                def scores(values):
                    return {'metrics':[{'si_sdr':v,'si_sar':5.,'target_gain':.5} for v in values]}
                baseline=scores([0.]*4)
                rows.append({'condition':condition,'scores':{'candidate':scores([-1.,5.,5.,5.] if condition=='wet_R4' else [0.]*4),**{k:baseline for k in ('initial_rev','clean','htdemucs','ensemble')}}})
        def stat(values):
            a=[v for v in values if v is not None]
            return {'median':float(np.median(a)),'ci':[min(a),max(a)]}
        result=summarize(rows,config,stat)
        self.assertTrue(result['gates']['primary'])
        self.assertFalse(result['gates']['wet_vocals'])
        self.assertFalse(result['passes'])

if __name__=='__main__':unittest.main()
