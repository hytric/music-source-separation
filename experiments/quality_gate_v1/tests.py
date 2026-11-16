import json,unittest
from pathlib import Path
import numpy as np
from core import split_songs,quality_label,choose_policy,fit_linear
GUIDE=Path('/run/media/iiplab/SSD2TB/code/26-2 반엘프/seperation/전체실험검수_2026-11-14/구현가이드')
class Contracts(unittest.TestCase):
 def test_split_order_independent_and_song_grouped(self):
  names=[str(i) for i in range(80)];a=split_songs(names,'salt');b=split_songs(names[::-1],'salt');self.assertEqual(a,b);self.assertEqual(list(a.values()).count('fit'),64)
 def test_all_label_fixtures(self):
  c=json.loads((GUIDE/'protocol.template.json').read_text())
  for r in json.loads((GUIDE/'contract_cases.json').read_text())['quality_label_cases']:
   self.assertEqual(quality_label(r['delta_si_sdr'],r['target_rms_fraction'],r['bass_gain_B'],r['bass_gain_H'],c),r['expected_label'])
 def test_all_calibration_fixtures(self):
  for r in json.loads((GUIDE/'contract_cases.json').read_text())['calibration_cases']:
   best,state=choose_policy(r['candidates']);self.assertEqual(best.get('policy',best.get('threshold')),r['expected_policy'])
   if 'expected_state' in r:self.assertEqual(state,r['expected_state'])
 def test_weighted_fit_and_no_state_mutation(self):
  c=json.loads((GUIDE/'protocol.template.json').read_text());x=np.array([[-3.],[-2.],[-1.],[1.],[2.],[3.]]);y=np.array([0,0,0,1,1,1]);w=np.full(6,1/6)
  m=fit_linear(x,y,w,c);self.assertGreater(m['beta'][1],0);self.assertEqual(m['mean'],[0.]);self.assertLess(m['iterations'],100)
  with self.assertRaises(ValueError):fit_linear(x,np.zeros(6),w,c)
if __name__=='__main__':unittest.main()
