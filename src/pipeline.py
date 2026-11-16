import argparse, hashlib, json, logging, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import bpm, bandpass, pos, protocol, normalize_clip
LOG=logging.getLogger('rppg')


def output(cfg):
    p=Path(cfg['output_path']).resolve();p.mkdir(parents=True,exist_ok=True);return p

def extract(video,cfg,dest,hands=False):
    import cv2
    from roi import Tracker
    cap=cv2.VideoCapture(str(video))
    source_fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or source_fps<=0:raise ValueError(f'Cannot decode video/fps: {video}')
    # Sequential decode on a uniform grid. Actual source selection times retained.
    rate=cfg['fps']; names=list(cfg['face_rois'])
    if hands: names += [f'{s}_{n}' for s in ['Left','Right'] for n in cfg['hand_rois']]
    data={n:{k:[] for k in ['patch','rgb','center','valid']} for n in names}
    times=[]; sources=[]; k=0; next_t=0.;tracker=Tracker(cfg); writer=None
    dest.mkdir(parents=True,exist_ok=True)
    if hands:LOG.info('ROI video preview is enabled=%s; inspect before interpretation',cfg['analysis']['export_roi_preview'])
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            t=k/source_fps;k+=1
            if hands and t>=cfg.get('max_input_seconds',float('inf')):break
            if t+1e-6<next_t:continue
            # Warn rather than silently upsample low frame-rate recordings.
            if source_fps<15:raise ValueError(f'{video}: source frame rate below 15 fps; acquisition unsuitable for default pipeline')
            rois,shapes=tracker.extract(frame,hands and bool(cfg.get('hand_rois')))
            while next_t<=t+1e-6:
                for n in names:
                    v=rois.get(n);data[n]['valid'].append(v is not None)
                    data[n]['patch'].append(v[0] if v else np.zeros((cfg['size'],cfg['size'],3),np.uint8))
                    data[n]['rgb'].append(v[1] if v else [np.nan]*3)
                    data[n]['center'].append(v[2] if v else [np.nan]*2)
                times.append(next_t);sources.append(t);next_t+=1/rate
                if hands and cfg['analysis']['export_roi_preview']:
                    if writer is None:
                        h,w=frame.shape[:2]; writer=cv2.VideoWriter(str(dest/'roi_tracking.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),rate,(w,h))
                        if not writer.isOpened():raise RuntimeError('Cannot write ROI preview')
                    for n,p in shapes.items():
                        cv2.polylines(frame,[p],True,(0,255,0),1)
                        cv2.putText(frame,n,tuple(p[0]),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,0),1)
                    writer.write(frame)
    finally:
        cap.release();tracker.close()
        if writer:writer.release()
    if not times:raise ValueError(f'No decoded frames: {video}')
    for n,d in data.items():
        np.savez_compressed(dest/f'{n}.npz',time=times,source_time=sources,**{k:np.asarray(v) for k,v in d.items()})
    (dest/'source.json').write_text(json.dumps({'video':str(video.resolve()),'source_fps':source_fps,'analysis_fps':rate,'samples':len(times)},indent=2))
    return dest

def prepare(cfg):
    root=Path(cfg['ubfc_path']).expanduser().resolve();dest=output(cfg)/'ubfc_cache';dest.mkdir(exist_ok=True)
    videos=sorted(root.rglob('vid.avi'))
    if not videos:raise ValueError('No vid.avi found under ubfc_path. Extract DATASET_2 archives first.')
    subjects=[]
    for video in videos:
        sid=video.parent.name
        if sid in subjects:raise ValueError(f'Duplicate subject folder name: {sid}')
        gtfile=video.parent/'ground_truth.txt'
        if not gtfile.exists():raise ValueError(f'Missing {gtfile}')
        cache=dest/sid
        LOG.info('Extracting UBFC %s',sid);extract(video,cfg,cache)
        from reference import align_reference
        with np.load(cache/f"{cfg['face_rois'][0]}.npz") as a:vt=a['time']
        np.savez(cache/'reference.npz',**align_reference(gtfile,vt,cache/'reference_diagnostics.json'))
        if cfg['evm']['enabled'] and cfg['evm'].get('ubfc_enabled',False) and sid.lower().replace('subject','') not in ['21','27']:
            evm(video,cache,cfg)
        subjects.append(sid)
    rng=random.Random(cfg['seed']);rng.shuffle(subjects)
    n=len(subjects);nt=int(n*cfg['train_fraction']);nv=max(1,int(n*cfg['validation_fraction']))
    if min(nt,n-nt-nv)<1:raise ValueError('Need enough participants for disjoint train/validation/test splits')
    split={'train':subjects[:nt],'validation':subjects[nt:nt+nv],'test':subjects[nt+nv:]}
    (output(cfg)/'splits.json').write_text(json.dumps(split,indent=2))
    return split


def train(cfg):
    import torch
    from torch.utils.data import DataLoader
    from streaming_data import VideoClips
    from model import CNNLSTM,correlation_loss
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    out=output(cfg);split=json.loads((out/'splits.json').read_text());length=int(cfg['fps']*cfg['clip_seconds'])
    tr=VideoClips(out/'ubfc_cache',split['train'],cfg['face_rois'],length,cfg['seed'],shuffle=True)
    va=VideoClips(out/'ubfc_cache',split['validation'],cfg['face_rois'],length,cfg['seed'],shuffle=False)
    if not len(tr) or not len(va):raise ValueError('No complete valid train/validation clips. Inspect ROI coverage.')
    device='cuda' if torch.cuda.is_available() else 'cpu';model=CNNLSTM().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate']);history=[];best=float('inf');bad=0
    for epoch in range(cfg['epochs']):
        vals=[];model.train();tr.set_epoch(epoch)
        for x,y in DataLoader(tr,batch_size=cfg['batch_size'],num_workers=0):
            opt.zero_grad();loss=correlation_loss(model(x.to(device)),y.to(device));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();vals.append(loss.item())
        model.eval();v=[]
        with torch.no_grad():
            for x,y in DataLoader(va,batch_size=cfg['batch_size'],num_workers=0):v.append(correlation_loss(model(x.to(device)),y.to(device)).item())
        row={'epoch':epoch+1,'train_loss':float(np.mean(vals)),'validation_loss':float(np.mean(v))};history.append(row);LOG.info('Epoch %s',row)
        if row['validation_loss']<best:
            best=row['validation_loss'];bad=0
            torch.save({'state_dict':model.state_dict(),'config':cfg,'split':split,'best_epoch':epoch+1},out/'cnn_lstm.pt')
        else:bad+=1
        pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
        if bad>=cfg['patience']:break
    fig,ax=plt.subplots();pd.DataFrame(history).set_index('epoch').plot(ax=ax);ax.set_ylabel('1 - waveform correlation');fig.savefig(out/'training_curve.png',dpi=160);plt.close(fig)


def load_model(cfg):
    import torch
    from model import CNNLSTM
    checkpoint=torch.load(cfg.get('model_path',output(cfg)/'cnn_lstm.pt'),map_location='cpu',weights_only=False)
    for key in ['fps','size','clip_seconds']:
        if cfg[key]!=checkpoint['config'][key]:raise ValueError(f'{key} differs from checkpoint: retrain or restore configuration')
    model=CNNLSTM();model.load_state_dict(checkpoint['state_dict']);model.eval();return model


def fill_short(x,valid,maxgap):
    valid=valid.copy();bad=np.where(~valid)[0]
    if not valid.any():return x.copy(),False
    groups=np.split(bad,np.where(np.diff(bad)>1)[0]+1)
    if any(len(g)>maxgap or (len(g) and (g[0]==0 or g[-1]==len(valid)-1)) for g in groups):return x.copy(),False
    a=x.copy();i=np.arange(len(a))
    for c in range(a.shape[1]):a[:,c]=np.interp(i,i[valid],a[valid,c])
    return a,True


def predict(a,model,cfg):
    import torch
    fs=cfg['fps'];length=int(fs*cfg['clip_seconds']);n=len(a['time']);result=np.full(n,np.nan)
    # Fixed non-overlapping model clips; each has an independent LSTM state.
    starts=list(range(0,n-length+1,length))
    if n>=length and (not starts or starts[-1]+length<n):starts.append(n-length)
    for start in starts:
        sl=slice(start,start+length)
        if not a['valid'][sl].all():continue
        with torch.no_grad():y=model(torch.from_numpy(normalize_clip(a['patch'][sl]))[None]).numpy()[0]
        if np.std(y)>1e-9:
            missing=~np.isfinite(result[sl]);chunk=result[sl];chunk[missing]=((y-y.mean())/y.std())[missing];result[sl]=chunk
    return result


def toolbox_pos(patches,fs,cfg):
    if not cfg['toolbox']['enabled']:return None
    root=Path(cfg['toolbox']['path']).resolve()
    if not (root/'unsupervised_methods/methods/POS_WANG.py').exists():raise ValueError('toolbox.enabled but no official POS_WANG.py at toolbox.path')
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    # Old toolbox revisions use np.mat (requires NumPy <2).
    from unsupervised_methods.methods.POS_WANG import POS_WANG
    return np.asarray(POS_WANG(patches,fs)).reshape(-1)


def analyse_cache(cache,segments,model,cfg,reference=None):
    fs=cfg['fps'];ac=cfg['analysis'];rows=[];waves=[]
    for file in sorted(cache.glob('*.npz')):
        if file.stem=='reference':continue
        a=np.load(file);t=a['time'];pred=predict(a,model,cfg)
        for si,seg in enumerate(segments):
            begin=seg['start']+ac['transition_exclude_seconds'];end=seg['end']-ac['transition_exclude_seconds']
            start=begin
            while start<end:
                stop=min(start+ac['window_seconds'],end);idx=np.where((t>=start)&(t<stop))[0];start=stop
                if len(idx)<ac['minimum_window_seconds']*fs:continue
                valid=a['valid'][idx];coverage=float(valid.mean());rgb=a['rgb'][idx];centers=a['center'][idx]
                rgb,filled=fill_short(rgb,valid,max(1,int(ac['maximum_gap_seconds']*fs)))
                motion=float(np.nanpercentile(np.linalg.norm(np.diff(centers,axis=0),axis=1),95)) if valid.sum()>1 else np.nan
                jump=float(np.nanmax(np.abs(np.diff(rgb,axis=0)))/255) if filled else np.nan
                row=dict(roi=file.stem,segment=si,label=seg['label'],pair=seg.get('pair',si//2+1),start_s=float(t[idx[0]]),end_s=float(t[idx[-1]]+1/fs),duration_s=len(idx)/fs,roi_coverage=coverage,motion=motion,brightness_jump=jump,model_bpm=np.nan,pos_bpm=np.nan,reference_bpm=np.nan,toolbox_pos_bpm=np.nan,concentration=np.nan,status='rejected',reason='')
                reasons=[]
                if coverage<ac['minimum_roi_coverage'] or not filled:reasons.append('missing_roi')
                if not np.isfinite(motion) or motion>ac['maximum_motion']:reasons.append('motion')
                if not np.isfinite(jump) or jump>ac['maximum_brightness_jump']:reasons.append('brightness_jump')
                if filled:
                    baseline=pos(rgb,fs);row['pos_bpm'],pq=bpm(baseline,fs)
                else:baseline=np.full(len(idx),np.nan)
                yy=pred[idx];interpolated,complete=fill_short(yy[:,None],np.isfinite(yy),max(1,int(ac['maximum_gap_seconds']*fs)))
                if not complete:reasons.append('missing_model_samples')
                else:
                    yy=interpolated[:,0];row['model_bpm'],row['concentration']=bpm(yy,fs)
                    if not np.isfinite(row['model_bpm']) or row['concentration']<ac['minimum_spectral_concentration']:reasons.append('weak_periodicity')
                    if np.isfinite(row['pos_bpm']) and abs(row['model_bpm']-row['pos_bpm'])>ac['maximum_method_difference_bpm']:reasons.append('method_disagreement')
                if reference is not None:
                    if reference['valid'][idx].all():row['reference_bpm'],_=bpm(reference['ppg'][idx],fs)
                    else:reasons.append('reference_outside_time_range')
                if cfg['toolbox']['enabled'] and valid.all():
                    tb=toolbox_pos(a['patch'][idx],fs,cfg);row['toolbox_pos_bpm'],_=bpm(tb,fs)
                row['reason']=';'.join(reasons);row['status']='accepted' if not reasons else 'rejected';rows.append(row)
                if ac['export_waveform_csv']:
                    for j,k in enumerate(idx):waves.append(dict(roi=file.stem,time_s=float(t[k]),label=seg['label'],model_wave=float(yy[j]) if complete else np.nan,pos_wave=float(baseline[j]),accepted=not reasons))
    df=pd.DataFrame(rows)
    if df.empty:raise ValueError('No analysis windows. Video too short for protocol/transition/window settings.')
    df.to_csv(cache/'windows.csv',index=False)
    if waves:pd.DataFrame(waves).to_csv(cache/'waveforms.csv',index=False)
    summary=[]
    for (roi,si),g in df.groupby(['roi','segment']):
        good=g[g.status=='accepted'];seg=segments[si];duration=seg['end']-seg['start'];sec=good.duration_s.sum()
        mean=float(np.average(good.model_bpm,weights=good.duration_s)) if sec else np.nan
        summary.append(dict(roi=roi,segment=si,label=seg['label'],pair=seg.get('pair',si//2+1),segment_duration_s=duration,valid_seconds=sec,coverage=sec/duration,average_bpm=mean,estimated_pulses_valid_duration=float((good.model_bpm*good.duration_s/60).sum()) if sec else np.nan,whole_segment_pulse_count=np.nan,interpretation='video_based_face_estimate; no blood-flow inference'))
    sm=pd.DataFrame(summary);sm.to_csv(cache/'segments.csv',index=False)
    if waves:
        wf=pd.DataFrame(waves);fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
        first=wf.roi.iloc[0];w=wf[wf.roi==first];w=w[w.time_s<w.time_s.min()+10]
        axes[0].plot(w.time_s,w.model_wave);axes[0].set_ylabel('CNN–LSTM (a.u.)')
        axes[1].plot(w.time_s,w.pos_wave);axes[1].set_ylabel('POS (a.u.)');axes[1].set_xlabel('Time (s)')
        fig.suptitle(f'{first}: first analysed 10 s; inspect windows.csv for acceptance')
        fig.tight_layout();fig.savefig(cache/'waveform_example.png',dpi=160);plt.close(fig)
    paired=[]
    for (roi,pair),g in sm.groupby(['roi','pair']):
        rest=g[g.label=='silence']
        for _,r in g[~g.label.isin(['silence','UNLABELLED'])].iterrows():
            base=rest.average_bpm.iloc[0] if len(rest)==1 else np.nan
            paired.append(dict(roi=roi,pair=pair,chant=r.label,silence_bpm=base,chant_bpm=r.average_bpm,delta_bpm=r.average_bpm-base))
    pd.DataFrame(paired,columns=['roi','pair','chant','silence_bpm','chant_bpm','delta_bpm']).to_csv(cache/'paired_changes.csv',index=False)
    video_summary=[]
    for roi,g in sm.groupby('roi'):
        sec=float(g.valid_seconds.sum());duration=float(g.segment_duration_s.sum())
        video_summary.append(dict(roi=roi,protocol_covered_seconds=duration,valid_seconds=sec,coverage=sec/duration if duration else 0,average_bpm=float(np.nansum(g.average_bpm*g.valid_seconds)/sec) if sec else np.nan,estimated_total_pulses_valid_duration=float(g.estimated_pulses_valid_duration.sum()) if sec else np.nan))
    pd.DataFrame(video_summary).to_csv(cache/'video_summary.csv',index=False)
    plot_results(df,sm,cache)
    return df,sm


def plot_results(df,sm,path):
    fig,ax=plt.subplots(figsize=(12,5))
    for roi,g in df.groupby('roi'):
        g=g.sort_values('start_s');values=g.model_bpm.where(g.status=='accepted');ax.plot((g.start_s+g.end_s)/2,values,'o-',label=roi)
    ax.set(xlabel='Time (s)',ylabel='Estimated pulse rate (BPM)',title='Accepted windows only; gaps indicate rejected/missing estimates')
    if ax.lines:ax.legend(fontsize=7,ncol=3)
    fig.tight_layout();fig.savefig(path/'pulse_rate_timeline.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5))
    for roi,g in sm.groupby('roi'):ax.plot(g.segment,g.average_bpm,'o-',label=roi)
    labels=sm.drop_duplicates('segment').sort_values('segment');ax.set_xticks(labels.segment,labels.label,rotation=30);ax.set_ylabel('Average estimated BPM')
    if ax.lines:ax.legend(fontsize=7,ncol=3)
    fig.tight_layout();fig.savefig(path/'condition_comparison.png',dpi=160);plt.close(fig)


def evaluate(cfg,model):
    out=output(cfg);split=json.loads((out/'splits.json').read_text());allrows=[]
    for sid in split['test']:
        cache=out/'ubfc_cache'/sid;ref=np.load(cache/'reference.npz');duration=len(ref['time'])/cfg['fps']
        df,_=analyse_cache(cache,[dict(start=0,end=duration,label='UBFC_test',pair=1)],model,cfg,ref)
        df['subject']=sid;allrows.append(df)
    df=pd.concat(allrows,ignore_index=True);df.to_csv(out/'ubfc_test_windows.csv',index=False)
    metrics=[]
    for roi,g in df.groupby('roi'):
        # Report unfiltered available estimates as well as heuristic-QC subset.
        for subset in ['all_available','accepted']:
            s=g if subset=='all_available' else g[g.status=='accepted'];s=s.dropna(subset=['model_bpm','reference_bpm'])
            e=s.model_bpm-s.reference_bpm
            metrics.append(dict(roi=roi,subset=subset,n_windows=len(s),n_subjects=s.subject.nunique(),mae_bpm=float(abs(e).mean()),rmse_bpm=float(np.sqrt(np.mean(e**2))),accepted_fraction=float((g.status=='accepted').mean())))
    pd.DataFrame(metrics).to_csv(out/'ubfc_metrics.csv',index=False)
    good=df.dropna(subset=['model_bpm','reference_bpm'])
    fig,ax=plt.subplots();ax.scatter(good.reference_bpm,good.model_bpm);ax.plot([42,180],[42,180],'k--');ax.set(xlabel='Reference PPG BPM',ylabel='CNN–LSTM BPM',title='Held-out UBFC participants: all available estimates');fig.tight_layout();fig.savefig(out/'ubfc_agreement.png',dpi=160);plt.close(fig)


def evm(video,dest,cfg):
    import cv2
    from scipy.signal import butter,sosfiltfilt
    ec=cfg['evm'];cap=cv2.VideoCapture(str(video));fs=cap.get(cv2.CAP_PROP_FPS);frames=[]
    try:
        for _ in range(int(fs*ec['max_seconds'])):
            ok,f=cap.read()
            if not ok:break
            # Gaussian pyramid level used for temporal colour amplification.
            f=cv2.resize(f,(320,240)).astype(np.float32)/255
            frames.append(cv2.pyrDown(cv2.pyrDown(f)))
    finally:cap.release()
    if len(frames)<max(30,fs*3):return
    tiny=np.array(frames);sos=butter(2,[ec['low_hz'],ec['high_hz']],fs=fs,btype='bandpass',output='sos');delta=sosfiltfilt(sos,tiny,axis=0)*ec['gain']
    cap=cv2.VideoCapture(str(video));writer=cv2.VideoWriter(str(dest/'evm_visualization_only.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fs,(320,240))
    if not writer.isOpened():raise RuntimeError('EVM VideoWriter failed')
    try:
        for d in delta:
            ok,f=cap.read()
            if not ok:break
            f=cv2.resize(f,(320,240)).astype(float)/255;d=cv2.resize(d,(320,240))
            enhanced=np.uint8(np.clip(f+d,0,1)*255)
            cv2.putText(enhanced,'EVM: VISUALIZATION ONLY',(5,15),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1);writer.write(enhanced)
    finally:cap.release();writer.release()


def analyse_inputs(cfg,model):
    p=Path(cfg['input_path']).expanduser().resolve()
    videos=[p] if p.is_file() else sorted(f for f in p.rglob('*') if f.suffix.lower() in ['.mp4','.avi','.mov','.mkv'])
    if not videos:raise ValueError(f'No input videos: {p}')
    out=output(cfg);allsummary=[];allvideos=[];errors=[]
    if p.is_dir() and (out==p or p in out.parents):raise ValueError('output_path must be outside input_path to avoid recursive re-analysis')
    for video in videos:
        name=video.stem+'_'+hashlib.sha256(str(video).encode()).hexdigest()[:8];dest=out/'user_results'/name
        try:
            LOG.info('Analysing %s',video)
            if cfg.get('expected_input_seconds'):
                import cv2
                cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);count=cap.get(cv2.CAP_PROP_FRAME_COUNT);cap.release()
                actual=count/fps if fps>0 else 0
                if abs(actual-cfg['expected_input_seconds'])>cfg.get('duration_tolerance_seconds',1):
                    raise ValueError(f'Expected four-minute constant-FPS video (240 +/-1 s), got {actual:.2f} s. Trim/align the input before prediction.')
            extract(video,cfg,dest,hands=True)
            a=np.load(dest/f"{cfg['face_rois'][0]}.npz");duration=len(a['time'])/cfg['fps'];segments=protocol(duration,cfg['protocol'])
            (dest/'protocol.json').write_text(json.dumps(segments,indent=2))
            _,sm=analyse_cache(dest,segments,model,cfg);sm['video']=str(video);allsummary.append(sm)
            vs=pd.read_csv(dest/'video_summary.csv');vs['video']=str(video);allvideos.append(vs)
            if cfg['evm']['enabled']:evm(video,dest,cfg)
        except Exception as ex:
            LOG.exception('Failed %s',video);errors.append({'video':str(video),'error':str(ex)})
    if allsummary:pd.concat(allsummary,ignore_index=True).to_csv(out/'all_video_results.csv',index=False)
    if allvideos:pd.concat(allvideos,ignore_index=True).to_csv(out/'all_video_summary.csv',index=False)
    (out/'input_errors.json').write_text(json.dumps(errors,indent=2))
    if errors:raise RuntimeError(f'{len(errors)} input(s) failed; see input_errors.json. Successful results retained.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='config.yaml');parser.add_argument('--stage',choices=['all','prepare','train','evaluate','analyse'],default='all');args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text());out=output(cfg)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(out/'run.log')])
    (out/'effective_config.yaml').write_text(yaml.safe_dump(cfg))
    # Fingerprint prevents accidentally using stale caches after edits.
    prepkeys=['ubfc_path','fps','size','face_rois','clip_seconds','seed','train_fraction','validation_fraction']
    signature=hashlib.sha256(json.dumps({k:cfg[k] for k in prepkeys},sort_keys=True).encode()).hexdigest();sigfile=out/'cache_signature.txt'
    if cfg['toolbox']['enabled']:
        from setup_toolbox import ensure_toolbox
        ensure_toolbox(cfg,out)
    if args.stage in ['all','prepare']:
        prepare(cfg);sigfile.write_text(signature)
    elif not sigfile.exists() or sigfile.read_text()!=signature:raise ValueError('Preparation config changed or missing: run --stage prepare, then train')
    if args.stage in ['all','train']:train(cfg)
    if args.stage in ['all','evaluate','analyse']:
        model=load_model(cfg)
        if args.stage in ['all','evaluate']:evaluate(cfg,model)
        if args.stage in ['all','analyse']:analyse_inputs(cfg,model)
    LOG.info('Completed stage %s; outputs: %s',args.stage,out)

if __name__=='__main__':main()
