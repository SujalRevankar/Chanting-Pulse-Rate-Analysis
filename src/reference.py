"""Dataset-2 reference reader with explicit diagnostics and bounded repair."""
import json,logging
from pathlib import Path
import numpy as np


def read_reference(path,diagnostics_path=None):
    path=Path(path);report={'file':str(path),'format':'UBFC Dataset 2: PPG / sensor HR / timestamps in seconds'}
    def save():
        if diagnostics_path:Path(diagnostics_path).write_text(json.dumps(report,indent=2,allow_nan=False))
        logging.info('Reference diagnostics: %s',report)
    def fail(message):
        report['error']=message;save()
        raise ValueError(message+' See reference_diagnostics.json; share that report and ground_truth.txt if unresolved.')
    lines=[l.strip() for l in path.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    report['nonempty_lines']=len(lines)
    if len(lines)!=3:fail(f'Expected exactly 3 nonempty lines, found {len(lines)}. Check Dataset-2 file format.')
    try:wave,hr,t=[np.asarray(l.split(),dtype=float) for l in lines]
    except ValueError:fail('Non-numeric token in ground_truth.txt; no data were silently truncated.')
    report.update(ppg_samples=len(wave),hr_samples=len(hr),timestamp_samples=len(t))
    if len(wave)!=len(t):fail(f'PPG/timestamp lengths differ ({len(wave)} versus {len(t)}); alignment cannot be safely inferred.')
    if len(t)<2:fail('Fewer than two reference samples.')
    finite=np.isfinite(t)&np.isfinite(wave)
    dropped=int((~finite).sum());report['nonfinite_ppg_or_time_samples']=dropped
    if dropped>max(1,.01*len(t)):fail('More than 1% of PPG/timestamp pairs are nonfinite; inspect source file.')
    if len(hr)==len(t):hr=hr[finite]
    else:
        report['sensor_hr_warning']='Sensor HR length differs; unused for waveform training/evaluation. Cached as NaN.'
        hr=np.full(int(finite.sum()),np.nan)
    wave,t=wave[finite],t[finite]
    if len(t)<2:fail('Not enough finite reference samples.')
    diff=np.diff(t);report['duplicate_adjacent_timestamps']=int((diff==0).sum());report['backward_timestamp_steps']=int((diff<0).sum())
    if (diff<0).any():
        where=np.where(diff<0)[0][:10];report['backward_examples']=[{'sample':int(i),'before':float(t[i]),'after':float(t[i+1])} for i in where]
        fail('Reference clock moves backward. The reader will not sort samples or invent a sampling rate.')
    # Repeated clock values are valid nondecreasing time labels. Coalesce corresponding
    # waveform samples at that recorded time; preserve the original time span.
    times,starts,counts=np.unique(t,return_index=True,return_counts=True)
    wave=np.add.reduceat(wave,starts)/counts
    validhr=np.isfinite(hr);sums=np.add.reduceat(np.where(validhr,hr,0),starts);n=np.add.reduceat(validhr.astype(float),starts)
    hr=np.divide(sums,n,out=np.full(len(times),np.nan),where=n>0)
    t=times-times[0]
    if len(t)<2:fail('All timestamps are identical; a pulse frequency cannot be established.')
    dt=np.diff(t);report.update(unique_timestamps=len(t),duration_seconds=float(t[-1]),median_sample_interval_seconds=float(np.median(dt)),largest_gap_seconds=float(dt.max()),repairs=[])
    if dropped:report['repairs'].append('Removed nonfinite PPG/time pairs; long resulting gaps excluded from labels')
    if report['duplicate_adjacent_timestamps']:report['repairs'].append('Averaged PPG samples sharing the exact same timestamp')
    if t[-1]<8:fail('Reference spans less than 8 seconds; verify timestamps are in seconds and the file is complete.')
    if np.median(dt)>.1:fail('Reference timestamps are too coarse (<10 Hz effective rate). Verify units and source file.')
    report['status']='ready';save()
    return t,wave,hr,report


def align_reference(path,video_times,diagnostics_path=None,max_gap_seconds=.25):
    t,wave,hr,report=read_reference(path,diagnostics_path)
    vt=np.asarray(video_times,float);valid=(vt>=t[0])&(vt<=t[-1])
    # Do not manufacture supervised labels across long reference gaps.
    right=np.clip(np.searchsorted(t,vt,side='right'),1,len(t)-1);left=right-1
    exact=np.isclose(vt,t[left],atol=1e-6,rtol=0)|np.isclose(vt,t[right],atol=1e-6,rtol=0)
    valid &= ((t[right]-t[left])<=max_gap_seconds)|exact
    duration=vt[-1]-vt[0] if len(vt)>1 else 0
    if duration and (t[-1]/duration<.75 or t[-1]/duration>1.25):logging.warning('Reference duration %.2f s differs substantially from video %.2f s; inspect units/start alignment',t[-1],duration)
    return dict(time=vt,ppg=np.interp(vt,t,wave),sensor_hr=np.interp(vt,t,hr),valid=valid)
