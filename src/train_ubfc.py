"""Automation 1: UBFC preparation -> CNN–LSTM training -> held-out evaluation."""
import argparse,json,logging
from pathlib import Path
import yaml
import pipeline

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ubfc',required=True,help='Extracted DATASET_2 root')
    p.add_argument('--output',default='training_output',help='Training outputs and saved model')
    p.add_argument('--config',default=str(Path(__file__).with_name('config.yaml')))
    p.add_argument('--epochs',type=int,help='Override training epochs')
    p.add_argument('--no-toolbox',action='store_true',help='Use built-in POS only; no Git needed')
    args=p.parse_args(argv)
    cfg=yaml.safe_load(Path(args.config).read_text())
    cfg.update(ubfc_path=str(Path(args.ubfc).expanduser().resolve()),output_path=str(Path(args.output).expanduser().resolve()))
    if args.epochs is not None:
        if args.epochs<1:p.error('--epochs must be positive')
        cfg['epochs']=args.epochs
    if args.no_toolbox:cfg['toolbox']['enabled']=False
    out=pipeline.output(cfg)
    if (out/'cnn_lstm.pt').exists():raise FileExistsError('This output already has a trained model. Choose a new --output directory to retain it.')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(out/'training.log')],force=True)
    (out/'training_config.yaml').write_text(yaml.safe_dump(cfg))
    if cfg['toolbox']['enabled']:
        from setup_toolbox import ensure_toolbox
        ensure_toolbox(cfg,out)
    pipeline.prepare(cfg)
    pipeline.train(cfg)
    model=pipeline.load_model(cfg)
    pipeline.evaluate(cfg,model)
    (out/'training_complete.json').write_text(json.dumps({'checkpoint':str(out/'cnn_lstm.pt'),'status':'trained_and_evaluated','accuracy_report':'ubfc_metrics.csv'},indent=2))
    print('\nTRAINING COMPLETE\nModel:',out/'cnn_lstm.pt','\nHeld-out accuracy:',out/'ubfc_metrics.csv')
    print('Next: python predict_videos.py --model "'+str(out/'cnn_lstm.pt')+'" --input "PATH/TO/VIDEO_OR_FOLDER" --output predictions')

if __name__=='__main__':main()
