"""Read one subject/ROI cache at a time, never retain the full dataset in RAM."""
from pathlib import Path
import random
import numpy as np
from torch.utils.data import IterableDataset
from core import normalize_clip

class VideoClips(IterableDataset):
    def __init__(self,cache_root,subject_ids,rois,length,seed=42,shuffle=False):
        self.root=Path(cache_root);self.ids=list(subject_ids);self.rois=list(rois)
        self.length=length;self.seed=seed;self.shuffle=shuffle;self.epoch=0
        # Only small metadata/index lists are retained. Image arrays are NOT loaded here.
        self.starts={}
        for sid in self.ids:
            with np.load(self.root/sid/'reference.npz') as ref:
                y=ref['ppg'];rv=ref['valid']
            for roi in self.rois:
                with np.load(self.root/sid/f'{roi}.npz') as a:valid=a['valid']
                starts=[]
                for s in range(0,len(valid)-length+1,length):
                    sl=slice(s,s+length)
                    if valid[sl].all() and rv[sl].all() and np.std(y[sl])>1e-8:starts.append(s)
                self.starts[(sid,roi)]=starts
    def __len__(self):return sum(map(len,self.starts.values()))
    def set_epoch(self,epoch):self.epoch=epoch
    def __iter__(self):
        rng=random.Random(self.seed+self.epoch);ids=self.ids.copy()
        if self.shuffle:rng.shuffle(ids)
        for sid in ids:
            with np.load(self.root/sid/'reference.npz') as ref:target=ref['ppg'].astype('float32')
            rois=self.rois.copy()
            if self.shuffle:rng.shuffle(rois)
            for roi in rois:
                starts=self.starts[(sid,roi)].copy()
                if not starts:continue
                if self.shuffle:rng.shuffle(starts)
                with np.load(self.root/sid/f'{roi}.npz') as a:patches=a['patch']
                # Only this video/ROI's small patches are held while yielding its clips.
                for s in starts:
                    y=target[s:s+self.length].copy();y=(y-y.mean())/(y.std()+1e-6)
                    yield normalize_clip(patches[s:s+self.length]),y
                del patches
            del target
