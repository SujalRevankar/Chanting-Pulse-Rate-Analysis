"""Add ONE UBFC Dataset-2 subject folder per invocation; resume automatically."""
import argparse,hashlib,json,logging,random
from pathlib import Path
import numpy as np
import torch,yaml
from torch.utils.data import DataLoader
from model import CNNLSTM,correlation_loss
from streaming_data import VideoClips
import pipeline

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject',required=True,help='ONE folder containing vid.avi and ground_truth.txt')
    p.add_argument('--state',default='incremental_training',help='Reuse this SAME directory for every subject')
    p.add_argument('--role',choices=['train','validation','test'],default='train',help='Role is permanent for a subject; held-out roles never update weights')
    p.add_argument('--passes',type=int,default=3,help='Epochs over all cached TRAIN subjects on this invocation')
    p.add_argument('--config',default=str(Path(__file__).with_name('config.yaml')))
    args=p.parse_args(argv)
    if args.passes<1:p.error('--passes must be positive')
    subject=Path(args.subject).expanduser().resolve();video=subject/'vid.avi';gt=subject/'ground_truth.txt'
    if not video.exists() or not gt.exists():raise ValueError('--subject must contain BOTH vid.avi and ground_truth.txt (Dataset 2)')
    out=Path(args.state).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    registry_file=out/'subjects.json';state_file=out/'incremental_state.pt'
    registry=json.loads(registry_file.read_text()) if registry_file.exists() else {}
    if state_file.exists():
        saved=torch.load(state_file,map_location='cpu',weights_only=False);cfg=saved['config']
    else:
        saved=None
        if (out/'cnn_lstm.pt').exists():raise ValueError('State folder contains a different training model. Choose a fresh incremental --state folder.')
        cfg=yaml.safe_load((out/'training_config.yaml').read_text()) if (out/'training_config.yaml').exists() else yaml.safe_load(Path(args.config).read_text())
        torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    cfg['output_path']=str(out);cfg['toolbox']['enabled']=False;cfg['evm']['ubfc_enabled']=False
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(out/'incremental.log')],force=True)
    sid=subject.name;signature=hashlib.sha256(gt.read_bytes()).hexdigest()
    if sid in registry:
        if registry[sid]['role']!=args.role:raise ValueError('Subject role cannot change; prevents train/test leakage')
        if registry[sid]['reference_sha256']!=signature:raise ValueError('Subject ID reused with different reference; choose unique subject folder names')
        logging.info('Reusing cached %s; no original-video re-extraction',sid)
    else:
        # Identical reference under another folder name must not cross role boundaries.
        for old in registry.values():
            if old['reference_sha256']==signature:raise ValueError('This reference was already added under another subject ID')
        cache=out/'ubfc_cache'/sid
        pipeline.extract(video,cfg,cache,hands=False)
        from reference import align_reference
        with np.load(cache/f"{cfg['face_rois'][0]}.npz") as a:vt=a['time']
        aligned=align_reference(gt,vt,cache/'reference_diagnostics.json')
        np.savez(cache/'reference.npz',**aligned)
        registry[sid]={'role':args.role,'reference_sha256':signature,'source_folder':str(subject)}
        registry_file.write_text(json.dumps(registry,indent=2))
    split={r:[s for s,x in registry.items() if x['role']==r] for r in ['train','validation','test']}
    (out/'splits.json').write_text(json.dumps(split,indent=2))
    (out/'training_config.yaml').write_text(yaml.safe_dump(cfg))
    model=CNNLSTM();optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    epochs=0
    if saved:
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer_state']);epochs=saved['completed_epochs']
    torch.set_num_threads(min(4,torch.get_num_threads()))
    device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
    for state in optimizer.state.values():
        for k,v in state.items():
            if torch.is_tensor(v):state[k]=v.to(device)
    length=int(cfg['fps']*cfg['clip_seconds'])
    history_file=out/'incremental_history.csv'
    import pandas as pd
    history=pd.read_csv(history_file).to_dict('records') if history_file.exists() else []
    if args.role=='train':
        train=VideoClips(out/'ubfc_cache',split['train'],cfg['face_rois'],length,cfg['seed'],shuffle=True)
        if not len(train):raise ValueError('No valid training clips; inspect ROI detection')
        valid=VideoClips(out/'ubfc_cache',split['validation'],cfg['face_rois'],length,cfg['seed'])
        for _ in range(args.passes):
            model.train();train.set_epoch(epochs);losses=[]
            for x,y in DataLoader(train,batch_size=1,num_workers=0):
                optimizer.zero_grad();loss=correlation_loss(model(x.to(device)),y.to(device));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(loss.item())
            model.eval();vl=[]
            with torch.no_grad():
                for x,y in DataLoader(valid,batch_size=1,num_workers=0):vl.append(correlation_loss(model(x.to(device)),y.to(device)).item())
            epochs+=1;row={'completed_epoch':epochs,'added_subject':sid,'train_subjects':len(split['train']),'train_loss':float(np.mean(losses)),'validation_loss':float(np.mean(vl)) if vl else np.nan};history.append(row)
            logging.info('%s',row)
            # Save at every completed pass; keep optimizer state for the next invocation.
            snapshot={'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer_state':optimizer.state_dict(),'completed_epochs':epochs,'config':cfg,'split':split}
            temp=out/'incremental_state.tmp';torch.save(snapshot,temp);temp.replace(state_file)
            torch.save({'state_dict':snapshot['state_dict'],'config':cfg,'split':split,'completed_epochs':epochs,'selection':'latest_incremental_epoch'},out/'cnn_lstm.pt')
            pd.DataFrame(history).to_csv(history_file,index=False)
        fig,ax=pipeline.plt.subplots();df=pd.DataFrame(history);ax.plot(df.completed_epoch,df.train_loss,label='training');ax.plot(df.completed_epoch,df.validation_loss,label='validation (if present)');ax.legend();ax.set(xlabel='Completed epoch',ylabel='1 - waveform correlation');fig.tight_layout();fig.savefig(out/'incremental_training_curve.png',dpi=160);pipeline.plt.close(fig)
    elif not saved:
        print('Held-out subject cached. Add a TRAIN subject next; no model exists yet.')
    # Do not use test metrics to select checkpoints or tune hyperparameters.
    if args.role=='test' and (out/'cnn_lstm.pt').exists() and split['test']:
        pipeline.evaluate(cfg,pipeline.load_model(cfg))
    status={'subjects_by_role':{r:len(v) for r,v in split.items()},'completed_epochs':epochs,'checkpoint_exists':(out/'cnn_lstm.pt').exists(),'held_out_test_available':bool(split['test']),'note':'No independent accuracy claim until held-out subjects are evaluated. Do not tune from repeated test inspection.'}
    (out/'incremental_status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2))
    print('Keep this state directory. Next invocation uses a NEW --subject path and the SAME --state path.')
    if status['checkpoint_exists']:print('Prediction model:',out/'cnn_lstm.pt')

if __name__=='__main__':main()
