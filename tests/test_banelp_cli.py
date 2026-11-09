import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiments.banelp.cli import HELP, SEQUENCE, main, _smoke_input

class CliTests(unittest.TestCase):
    def test_synthetic_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(__file__).parents[1]; artifact=Path(tmp)/'artifacts'; runtime=Path(tmp)/'runtime'
            self.assertEqual(main(['validate','--runtime-dir',str(runtime),'--artifact-dir',str(artifact),'--config',str(root/'experiments/banelp/config.json'),'--synthetic']),0)
    def test_shipped_selection_policy_is_confirmed(self):
        config=json.loads((Path(__file__).parents[1]/'experiments/banelp/config.json').read_text())
        self.assertEqual(config['selection'],{'sar_policy_status':'confirmed','sar_floor_db':-0.5,'minimum_sdr_gain_db':0.2,
                                              'baseline_train_data_policy':'allow-unknown-with-disclosure','model_candidates':[],
                                              'baseline_decisions':[{'model_id':'htdemucs','sha256':'8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4','demucs_signature':'955717e8',
                                                                     'decision':'use-with-disclosure (training data unverified)','decided_by':'user','date':'2026-11-09'}]})
    def test_seal_refuses_pending_sar_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(__file__).parents[1]; config=json.loads((root/'experiments/banelp/config.json').read_text())
            config['selection']={'sar_policy_status':'pending-owner-confirmation','sar_floor_db':None,'minimum_sdr_gain_db':0.2}
            path=Path(tmp)/'config.json'; path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'owner confirms SAR policy'): main(['seal','--runtime-dir',tmp+'/runtime','--artifact-dir',tmp+'/artifacts','--config',str(path)])
    def test_lock_policy_is_in_the_documented_order(self):
        self.assertTrue(SEQUENCE.endswith('smoke -> lock-policy -> develop -> seal -> evaluate'))
        self.assertIn('lock-policy',HELP)
    def test_develop_before_lock_is_refused_without_touching_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(__file__).parents[1]
            with self.assertRaisesRegex(FileNotFoundError,'lock-policy'):
                main(['develop','--runtime-dir',tmp+'/runtime','--artifact-dir',tmp+'/artifacts','--config',str(root/'experiments/banelp/config.json')])
    def test_smoke_slices_fixed_thirty_second_cache(self):
        x=np.arange(2*1_323_000,dtype=np.float32).reshape(2,-1)
        self.assertTrue(np.array_equal(_smoke_input(x,5,44100),x[:,:220500]))
        self.assertEqual(_smoke_input(x,30,44100).shape,(2,1_323_000))
        with self.assertRaises(ValueError): _smoke_input(x[:,:220500],5,44100)
