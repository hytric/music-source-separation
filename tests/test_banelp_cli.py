import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiments.banelp.cli import main, _smoke_input

class CliTests(unittest.TestCase):
    def test_synthetic_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(__file__).parents[1]; artifact=Path(tmp)/'artifacts'; runtime=Path(tmp)/'runtime'
            self.assertEqual(main(['validate','--runtime-dir',str(runtime),'--artifact-dir',str(artifact),'--config',str(root/'experiments/banelp/config.json'),'--synthetic']),0)
    def test_seal_waits_for_owner_sar_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(__file__).parents[1]
            with self.assertRaises(ValueError): main(['seal','--runtime-dir',tmp+'/runtime','--artifact-dir',tmp+'/artifacts','--config',str(root/'experiments/banelp/config.json')])
    def test_smoke_slices_fixed_thirty_second_cache(self):
        x=np.arange(2*1_323_000,dtype=np.float32).reshape(2,-1)
        self.assertTrue(np.array_equal(_smoke_input(x,5,44100),x[:,:220500]))
        self.assertEqual(_smoke_input(x,30,44100).shape,(2,1_323_000))
        with self.assertRaises(ValueError): _smoke_input(x[:,:220500],5,44100)
