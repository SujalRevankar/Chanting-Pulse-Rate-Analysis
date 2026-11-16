import sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from streaming_data import VideoClips

class Streaming(unittest.TestCase):
    def test_patches_loaded_only_as_each_video_is_consumed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for sid in ['A','B']:
                p=root/sid;p.mkdir()
                np.savez(p/'reference.npz',ppg=np.sin(np.arange(120)),valid=np.ones(120,bool))
                np.savez(p/'forehead.npz',valid=np.ones(120,bool),patch=np.full((120,32,32,3),100,np.uint8))
            original=np.load;reads=[]
            class Proxy:
                def __init__(self,path):self.path=Path(path);self.a=original(path)
                def __enter__(self):return self
                def __exit__(self,*args):self.a.close()
                def __getitem__(self,key):
                    if key=='patch':reads.append(self.path.parent.name)
                    return self.a[key]
            with patch('streaming_data.np.load',side_effect=Proxy):
                data=VideoClips(root,['A','B'],['forehead'],30,shuffle=False)
                self.assertEqual(reads,[])
                self.assertEqual(len(data),8)
                it=iter(data)
                for _ in range(4):next(it)
                self.assertEqual(reads,['A'])
                next(it);self.assertEqual(reads,['A','B'])
                it.close()

if __name__=='__main__':unittest.main()
