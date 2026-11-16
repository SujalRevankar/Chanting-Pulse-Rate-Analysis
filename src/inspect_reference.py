"""Inspect a Dataset-2 ground truth file without video extraction or training."""
import argparse,json
from pathlib import Path
from reference import read_reference
p=argparse.ArgumentParser();p.add_argument('--subject',required=True);a=p.parse_args()
report=Path('reference_diagnostics.json').resolve()
try:
    read_reference(Path(a.subject)/'ground_truth.txt',report)
finally:
    if report.exists():print(report.read_text());print('Diagnostic report:',report)
