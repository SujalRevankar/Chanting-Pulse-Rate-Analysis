"""Fetch a pinned official toolbox checkout, record provenance; no source edits."""
import json,subprocess
from pathlib import Path
REVISION='b7500b848f84ad7f86e277b4612563b69f4f88f9'
URL='https://github.com/ubicomplab/rPPG-Toolbox.git'

def ensure_toolbox(cfg,out):
    path=Path(cfg['toolbox']['path']).expanduser().resolve()
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        try:
            subprocess.run(['git','clone','--no-checkout',URL,str(path)],check=True)
            subprocess.run(['git','-C',str(path),'checkout',REVISION],check=True)
        except FileNotFoundError as e:raise RuntimeError('Install Git, or disable toolbox in config.yaml') from e
    if not (path/'unsupervised_methods/methods/POS_WANG.py').exists():raise ValueError(f'Not a toolbox checkout: {path}')
    rev=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    dirty=bool(subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True).strip())
    (out/'toolbox_provenance.json').write_text(json.dumps(dict(url=URL,commit=rev,dirty=dirty,path=str(path)),indent=2))
    return path
