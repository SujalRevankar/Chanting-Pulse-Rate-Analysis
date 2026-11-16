import sys,tempfile,types,json,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,cv2,yaml,torch
import train_incremental

class Incremental(unittest.TestCase):
    def test_resume_two_separate_subject_paths(self):
        torch.set_num_threads(2)
        class Tracker:
            def __init__(self,cfg):pass
            def close(self):pass
            def extract(self,bgr,with_hands=False):
                a=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(32,32))
                return {'forehead':(a,a.mean((0,1)),np.array([.5,.5]))},{}
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'src/config.yaml').read_text());cfg.update(fps=15,clip_seconds=4,face_rois=['forehead']);cfg['toolbox']['enabled']=False
            conf=d/'test.yaml';conf.write_text(yaml.safe_dump(cfg));state=d/'state'
            for k in [1,2]:
                p=d/f'subject{k}';p.mkdir();freq=1+.1*k
                w=cv2.VideoWriter(str(p/'vid.avi'),cv2.VideoWriter_fourcc(*'MJPG'),15,(32,32))
                for j in range(180):
                    a=np.full((32,32,3),100,np.uint8);a[:,:,1]=120+4*np.sin(2*np.pi*freq*j/15);w.write(a)
                w.release();t=np.arange(720)/60
                (p/'ground_truth.txt').write_text('\n'.join(' '.join(map(str,x)) for x in [np.sin(2*np.pi*freq*t),np.full(720,freq*60),t]))
            with patch.dict(sys.modules,{'roi':types.SimpleNamespace(Tracker=Tracker)}):
                train_incremental.main(['--subject',str(d/'subject1'),'--state',str(state),'--passes','1','--config',str(conf)])
                first=torch.load(state/'incremental_state.pt',weights_only=False)
                train_incremental.main(['--subject',str(d/'subject2'),'--state',str(state),'--passes','1','--config',str(conf)])
                second=torch.load(state/'incremental_state.pt',weights_only=False)
                self.assertEqual(second['completed_epochs'],2);self.assertEqual(len(second['split']['train']),2)
                self.assertTrue(any(not torch.equal(first['state_dict'][k],second['state_dict'][k]) for k in first['state_dict']))
                a=next(iter(first['optimizer_state']['state'].values()))['step'];b=next(iter(second['optimizer_state']['state'].values()))['step'];self.assertGreater(b,a)
                with self.assertRaises(ValueError):train_incremental.main(['--subject',str(d/'subject1'),'--state',str(state),'--role','test'])

if __name__=='__main__':unittest.main()
