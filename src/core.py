"""Signal processing and protocol utilities. No neural or video dependencies."""
import numpy as np
from scipy import signal

def bandpass(x, fs):
    x=np.asarray(x,float)
    return signal.sosfiltfilt(signal.butter(3,[0.7,3.0],fs=fs,btype='bandpass',output='sos'), signal.detrend(x))

def bpm(x, fs):
    if len(x)<fs*8 or not np.isfinite(x).all() or np.std(x)<1e-9:
        return np.nan, 0.
    y=bandpass(x,fs)
    f,p=signal.periodogram(y,fs=fs,window='hann',nfft=max(4096,len(y)))
    mask=(f>=0.7)&(f<=3)
    fp,pp=f[mask],p[mask]; peak=fp[np.argmax(pp)]
    # Spectral concentration is a heuristic quality score, NOT clinical confidence.
    concentration=pp[np.abs(fp-peak)<0.1].sum()/max(pp.sum(),1e-12)
    return float(60*peak),float(concentration)

def pos(rgb, fs):
    rgb=np.asarray(rgb,float); out=np.zeros(len(rgb)); win=int(np.ceil(fs*1.6))
    for end in range(win,len(rgb)+1):
        a=rgb[end-win:end]; c=a/np.maximum(a.mean(0),1e-6)-1
        s1=c[:,1]-c[:,2]; s2=-2*c[:,0]+c[:,1]+c[:,2]
        h=s1+np.std(s1)/max(np.std(s2),1e-9)*s2
        out[end-win:end]+=h-h.mean()
    return bandpass(out,fs)

def protocol(duration,cfg):
    if cfg.get('segments'):
        seg=[dict(s) for s in cfg['segments']]
        prev=0
        for s in seg:
            if s['start']<prev or s['end']<=s['start']: raise ValueError('Invalid/overlapping protocol segments')
            prev=s['end']
        return [dict(s,end=min(s['end'],duration)) for s in seg if s['start']<duration]
    block=cfg['block_seconds']; chants=cfg['chant_order']; out=[]
    for k in range(int(np.ceil(duration/block))):
        ci=k//2
        label='silence' if k%2==0 else (chants[ci] if ci<len(chants) else 'UNLABELLED')
        out.append(dict(start=k*block,end=min((k+1)*block,duration),label=label,pair=ci+1))
    return out

def normalize_clip(x):
    x=x.astype(np.float32)/255.
    # Per-pixel/channel temporal normalization for rate extraction, not amplitude analysis.
    return (x-x.mean(axis=0,keepdims=True))/(x.std(axis=0,keepdims=True)+0.01)
