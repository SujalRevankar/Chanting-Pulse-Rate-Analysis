"""Engineering smoke check with SYNTHETIC signals and mocked ROI detector.
NOT participant data, not an accuracy benchmark. Temporary outputs are deleted.
"""
import sys,tempfile,types,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,cv2,yaml,torch
import pipeline
from model import CNNLSTM,correlation_loss

def main():
    torch.set_num_threads(2)
    x=torch.randn(2,60,32,32,3);m=CNNLSTM();y=m(x)
    assert y.shape==(2,60)
    loss=correlation_loss(y,torch.randn(2,60));loss.backward()
    assert all(p.grad is not None for p in m.parameters())
    class FakeTracker:
        def __init__(self,cfg):self.cfg=cfg
        def close(self):pass
        def extract(self,bgr,with_hands=True):
            patch=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(32,32))
            v=(patch,patch.mean((0,1)),np.array([.5,.5]))
            names=self.cfg['face_rois']+(['Left_palm_center'] if with_hands else [])
            return {n:v for n in names},{}
    sys.modules['roi']=types.SimpleNamespace(Tracker=FakeTracker)
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);ubfc=root/'ubfc';inp=root/'inputs';inp.mkdir()
        def video(path,frequency):
            w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),15,(64,64));assert w.isOpened()
            for k in range(180):
                s=np.sin(2*np.pi*frequency*k/15);f=np.zeros((64,64,3),np.uint8);f[:]=[90+2*s,120+5*s,100+s];w.write(f)
            w.release()
        for k in range(6):
            p=ubfc/f'subject{k}';p.mkdir(parents=True);freq=1.1+.05*k;video(p/'vid.avi',freq)
            t=np.arange(720)/60;wave=np.sin(2*np.pi*freq*t)
            (p/'ground_truth.txt').write_text('\n'.join(' '.join(map(str,z)) for z in [wave,np.full(len(t),freq*60),t]))
        video(inp/'SYNTHETIC_NOT_STUDY_DATA.avi',1.2)
        cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'src/config.yaml').read_text())
        cfg.update(ubfc_path=str(ubfc),input_path=str(inp),output_path=str(root/'outputs'),fps=15,clip_seconds=4,epochs=1,face_rois=['forehead'],hand_rois=['palm_center'])
        cfg['toolbox']['enabled']=False
        cfg['evm']['ubfc_enabled']=False
        cfg['protocol'].update(block_seconds=6,chant_order=['Ha'])
        cfg['analysis'].update(window_seconds=12,minimum_window_seconds=8,transition_exclude_seconds=0,minimum_spectral_concentration=0,maximum_method_difference_bpm=200)
        pipeline.prepare(cfg);pipeline.train(cfg);model=pipeline.load_model(cfg);pipeline.evaluate(cfg,model)
        # Input protocol needs >=8s windows, so use 12s silence for functional check.
        cfg['protocol']['block_seconds']=60
        pipeline.analyse_inputs(cfg,model)
        out=Path(cfg['output_path']);assert (out/'cnn_lstm.pt').exists();assert (out/'ubfc_metrics.csv').exists()
        dest=next((out/'user_results').iterdir());assert (dest/'segments.csv').exists();assert (dest/'evm_visualization_only.mp4').stat().st_size>0
        assert (dest/'pulse_rate_timeline.png').exists()
        print('PASS: model forward/backward, synthetic video decode, prepare, train, checkpoint, evaluate, input analysis, CSV/PNG/MP4/EVM exports.')
        # Directly test condition pairing and duration arithmetic with enough data.
        cache=root/'pair_test';cache.mkdir();n=900;t=np.arange(n)/15;ppg=np.sin(2*np.pi*1.2*t)
        patches=np.zeros((n,32,32,3),np.uint8)
        for k in range(n):patches[k]=[100+ppg[k],120+4*ppg[k],90+ppg[k]]
        np.savez(cache/'forehead.npz',time=t,patch=patches,rgb=patches.mean((1,2)),center=np.full((n,2),.5),valid=np.ones(n,bool))
        class KnownWave:
            def __call__(self,x):return torch.tensor(np.sin(2*np.pi*1.2*np.arange(x.shape[1])/15)[None],dtype=torch.float32)
        # Use 10s clips so phase is continuous for 1.2Hz test wave.
        cfg['clip_seconds']=10
        df,sm=pipeline.analyse_cache(cache,[dict(start=0,end=30,label='silence',pair=1),dict(start=30,end=60,label='Ha',pair=1)],KnownWave(),cfg)
        assert len(sm)==2
        for _,row in sm.iterrows():
            assert abs(row.average_bpm-72)<1
            assert abs(row.estimated_pulses_valid_duration-row.average_bpm*row.valid_seconds/60)<1e-6
        pairs=__import__('pandas').read_csv(cache/'paired_changes.csv');assert abs(pairs.delta_bpm.iloc[0])<1
        print('PASS: synthetic known 72 BPM, paired protocol delta, weighted summaries and pulse-count arithmetic.')

if __name__=='__main__':main()
