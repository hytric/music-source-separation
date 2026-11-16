import importlib.util,unittest,json
from pathlib import Path
import numpy as np
from core import fit,predict,features

class SelectorTests(unittest.TestCase):
    def test_training_standardization_and_separation(self):
        x=np.array([[-3,0],[-2,1],[-1,0],[1,0],[2,1],[3,0]],float);y=np.array([0,0,0,1,1,1])
        model=fit(x,y);np.testing.assert_allclose(model['mean'],x.mean(0))
        before=json.dumps(model,sort_keys=True);p=predict(x,model)
        self.assertLess(p[0],.2);self.assertGreater(p[-1],.8)
        predict(np.array([[100.,-100.]]),model);self.assertEqual(before,json.dumps(model,sort_keys=True))
    def test_audio_feature_gain_and_channel_invariance(self):
        root=Path(__file__).resolve().parent.parent
        path=next(p for p in (root/'입력전처리검증_2026-11-13/run.py',root/'frontend_validation/run.py') if p.exists())
        spec=importlib.util.spec_from_file_location('selector_test_front',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        wave=np.random.default_rng(9).normal(0,.03,(2,26*44100)).astype(np.float32)
        a=features(wave,mod.stft);b=features(wave*3,mod.stft);c=features(wave[::-1].copy(),mod.stft)
        self.assertEqual(a.shape,(33,));np.testing.assert_allclose(a,b,atol=1e-4,rtol=1e-4);np.testing.assert_allclose(a,c,atol=1e-4,rtol=1e-4)
    def test_silence_and_bad_shape(self):
        def stft(x):return np.zeros((2,2049,20),np.complex64)
        np.testing.assert_array_equal(features(np.zeros((2,26*44100),np.float32),stft),np.zeros(33))
        with self.assertRaises(ValueError):features(np.zeros((1,26*44100)),stft)

if __name__=='__main__':unittest.main()
