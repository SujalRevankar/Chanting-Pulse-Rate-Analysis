"""Command separation/contract tests; real waveform accuracy is not asserted."""
import copy,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,cv2,torch,yaml
import train_ubfc,predict_videos,pipeline
from model import CNNLSTM

class Automations(unittest.TestCase):
    def test_prediction_independent_of_ubfc(self):
        torch.set_num_threads(2)
        cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'src/config.yaml').read_text())
        cfg.update(fps=15,clip_seconds=4,face_rois=['forehead'],hand_rois=['palm_center'],ubfc_path='/THIS/DOES/NOT/EXIST')
        cfg['toolbox']['enabled']=False;cfg['evm']['enabled']=False
        class FakeTracker:
            def __init__(self,cfg):pass
            def close(self):pass
            def extract(self,bgr,with_hands=True):
                a=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(32,32));v=(a,a.mean((0,1)),np.array([.5,.5]))
                return {'forehead':v,'Left_palm_center':v},{}
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);model=d/'saved_model.pt';torch.save({'state_dict':CNNLSTM().state_dict(),'config':cfg},model)
            video=d/'SYNTHETIC_NOT_PARTICIPANT_DATA.avi';w=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),15,(32,32));self.assertTrue(w.isOpened())
            for k in range(3600):
                s=np.sin(2*np.pi*1.2*k/15);a=np.zeros((32,32,3),np.uint8);a[:]=[90+s,120+4*s,100+s];w.write(a)
            w.release()
            out=d/'prediction'
            with patch.dict(sys.modules,{'roi':types.SimpleNamespace(Tracker=FakeTracker)}),patch.object(pipeline,'train',side_effect=AssertionError('Prediction must never train')),patch.object(pipeline,'prepare',side_effect=AssertionError('Prediction must never read UBFC')):
                predict_videos.main(['--model',str(model),'--input',str(video),'--output',str(out),'--no-evm','--no-toolbox'])
            meta=json.loads((out/'prediction_metadata.json').read_text());self.assertFalse(meta['model_is_retrained'])
            import pandas as pd
            results=pd.read_csv(out/'all_video_results.csv')
            one=results[results.roi=='forehead'].sort_values('segment')
            self.assertEqual(one.label.tolist(),['silence','Ha','silence','Ya'])
            self.assertTrue((out/'all_video_summary.csv').exists())
            cache=next((out/'user_results').iterdir());self.assertTrue((cache/'paired_changes.csv').exists())
            self.assertFalse((out/'ubfc_cache').exists())
    def test_train_wrapper_does_not_analyse_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            def fake_train(cfg):
                (Path(cfg['output_path'])/'cnn_lstm.pt').write_bytes(b'mock checkpoint only for command test')
            with patch.object(pipeline,'prepare') as prep,patch.object(pipeline,'train',side_effect=fake_train),patch.object(pipeline,'load_model',return_value=object()),patch.object(pipeline,'evaluate') as evaluate,patch.object(pipeline,'analyse_inputs',side_effect=AssertionError('Training must not analyse user inputs')):
                train_ubfc.main(['--ubfc',str(Path(d)/'UBFC'),'--output',str(Path(d)/'training'),'--no-toolbox','--epochs','1'])
                self.assertEqual(prep.call_count,1);self.assertEqual(evaluate.call_count,1)
                self.assertTrue((Path(d)/'training/training_complete.json').exists())

if __name__=='__main__':unittest.main()
