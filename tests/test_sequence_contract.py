"""Sequence metadata/label contract without Torch or participant videos."""
import importlib.util,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
SRC=Path(__file__).resolve().parents[1]/'src'
sys.path.insert(0,str(SRC))
from core import protocol

class SequenceContract(unittest.TestCase):
 def test_order_metadata_and_monkey_patch_restoration(self):
  for order,labels in [('HA_YA',['silence','Ha','silence','Ya']),('YA_HA',['silence','Ya','silence','Ha'])]:
   with tempfile.TemporaryDirectory() as directory:
    out=Path(directory);(out/'prediction_metadata.json').write_text('{}')
    observed={}
    def original_analyse(cfg,model):observed.update(cfg);return 'ok'
    fake=types.ModuleType('pipeline');fake.protocol=protocol;fake.analyse_inputs=original_analyse
    with patch.dict(sys.modules,{'pipeline':fake}):
     spec=importlib.util.spec_from_file_location('sequence_under_test',SRC/'predict_sequence.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
     def fake_predict(*args,**kwargs):
      cfg={'output_path':str(out),'protocol':{'segments':[dict(start=i*60,end=(i+1)*60,label='original',pair=i//2+1) for i in range(4)]}}
      segments=fake.protocol(240,cfg['protocol']);self.assertEqual([s['label'] for s in segments],labels)
      self.assertEqual(fake.analyse_inputs(cfg,None),'ok')
     with patch.object(sys,'argv',['predict_sequence.py','--order',order]),patch.object(mod.runpy,'run_path',side_effect=fake_predict):mod.main()
    self.assertIs(fake.protocol,protocol);self.assertIs(fake.analyse_inputs,original_analyse)
    metadata=json.loads((out/'prediction_metadata.json').read_text());self.assertEqual([s['label'] for s in metadata['protocol']],labels)
    self.assertEqual([s['label'] for s in observed['protocol']['segments']],labels)

if __name__=='__main__':unittest.main()
