"""Automation 2: saved checkpoint -> four-minute video prediction (NO training)."""
import argparse,copy,json,logging
from pathlib import Path
import torch,yaml
import pipeline

SEGMENTS=[{'start':0,'end':60,'label':'silence','pair':1},
          {'start':60,'end':120,'label':'Ha','pair':1},
          {'start':120,'end':180,'label':'silence','pair':2},
          {'start':180,'end':240,'label':'Ya','pair':2}]

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,help='cnn_lstm.pt saved by train_ubfc.py')
    p.add_argument('--input',required=True,help='A face video or folder containing face videos')
    p.add_argument('--output',default='predictions')
    p.add_argument('--no-toolbox',action='store_true')
    p.add_argument('--no-evm',action='store_true')
    args=p.parse_args(argv)
    checkpoint=Path(args.model).expanduser().resolve()
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=copy.deepcopy(saved['config'])
    cfg.update(input_path=str(Path(args.input).expanduser().resolve()),output_path=str(Path(args.output).expanduser().resolve()),model_path=str(checkpoint),expected_input_seconds=240,duration_tolerance_seconds=1.0,max_input_seconds=240)
    cfg['protocol']={'segments':copy.deepcopy(SEGMENTS)}
    cfg['hand_rois']=[]
    cfg['analysis']['export_roi_preview']=False
    cfg['evm']['enabled']=False
    if args.no_toolbox:cfg['toolbox']['enabled']=False
    if args.no_evm:cfg['evm']['enabled']=False
    out=pipeline.output(cfg)
    if out==checkpoint.parent:raise ValueError('--output must differ from the training/checkpoint directory')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(out/'prediction.log')],force=True)
    (out/'prediction_config.yaml').write_text(yaml.safe_dump(cfg))
    (out/'prediction_metadata.json').write_text(json.dumps({'checkpoint':str(checkpoint),'model_is_retrained':False,'preprocessing_from_checkpoint':True,'protocol':SEGMENTS,'reference_measurements_available':False},indent=2))
    if cfg['toolbox']['enabled']:
        from setup_toolbox import ensure_toolbox
        ensure_toolbox(cfg,out)
    model=pipeline.load_model(cfg)
    pipeline.analyse_inputs(cfg,model)
    print('\nPREDICTION COMPLETE\nPer-video:',out/'all_video_summary.csv','\nPer-condition:',out/'all_video_results.csv')
    print('Inspect each roi_tracking.mp4 and windows.csv before interpreting estimates. No reference sensor means no accuracy score on these inputs.')

if __name__=='__main__':main()
