import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from core import bpm,protocol,pos
from pipeline import fill_short

class Calculations(unittest.TestCase):
    def test_rate(self):
        fs=30;t=np.arange(900)/fs
        rate,q=bpm(np.sin(2*np.pi*1.2*t),fs)
        self.assertAlmostEqual(rate,72,delta=.5);self.assertGreater(q,.8)
    def test_flat(self):self.assertTrue(np.isnan(bpm(np.ones(900),30)[0]))
    def test_protocol(self):
        s=protocol(245,dict(block_seconds=60,chant_order=['Ha','Ya']))
        self.assertEqual([x['label'] for x in s],['silence','Ha','silence','Ya','silence']);self.assertEqual(s[-1]['end'],245)
    def test_missing(self):
        a=np.ones((100,3));v=np.ones(100,bool);v[40:42]=False;a[~v]=np.nan
        x,ok=fill_short(a,v,3);self.assertTrue(ok);self.assertTrue(np.isfinite(x).all())
        v[40:50]=False;self.assertFalse(fill_short(a,v,3)[1])
    def test_invalid_protocol(self):
        with self.assertRaises(ValueError):protocol(100,{'segments':[{'start':30,'end':20,'label':'Ha'}]})
    def test_pos(self):
        t=np.arange(900)/30;s=np.sin(2*np.pi*1.2*t)
        rgb=np.stack([100+.4*s,120+1.5*s,90+.2*s],1)
        rate,_=bpm(pos(rgb,30),30);self.assertAlmostEqual(rate,72,delta=1)

if __name__=='__main__':unittest.main()
