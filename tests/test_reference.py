import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from reference import read_reference,align_reference

class Reference(unittest.TestCase):
    def make(self,p,w,t,hr=None):
        hr=np.ones(len(t))*72 if hr is None else hr
        p.write_text('\n'.join(' '.join(map(str,x)) for x in [w,hr,t]))
    def test_duplicate_times_coalesced(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'gt.txt';t=np.repeat(np.arange(600)/60,2);w=np.sin(2*np.pi*1.2*t)
            self.make(p,w,t);tt,ww,hr,r=read_reference(p)
            self.assertEqual(len(tt),600);self.assertEqual(r['duplicate_adjacent_timestamps'],600);np.testing.assert_allclose(ww,w[::2])
    def test_backward_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'gt.txt';t=np.arange(600)/60;t[100]=0
            self.make(p,np.sin(t),t)
            with self.assertRaisesRegex(ValueError,'clock moves backward'):read_reference(p)
    def test_misaligned_lengths_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'gt.txt';self.make(p,np.ones(5),np.arange(600)/60)
            with self.assertRaisesRegex(ValueError,'lengths differ'):read_reference(p)
    def test_unused_hr_length_can_differ(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'gt.txt';t=np.arange(600)/60;self.make(p,np.sin(t),t,np.ones(10))
            _,_,hr,r=read_reference(p);self.assertTrue(np.isnan(hr).all())
    def test_long_reference_gap_not_interpolated_as_valid(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'gt.txt';t=np.r_[np.arange(300)/60,np.arange(360,660)/60]
            self.make(p,np.sin(t),t);aligned=align_reference(p,np.array([2.,5.5,8.]))
            self.assertEqual(aligned['valid'].tolist(),[True,False,True])

if __name__=='__main__':unittest.main()
