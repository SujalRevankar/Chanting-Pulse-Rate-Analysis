import cv2
import mediapipe as mp
import numpy as np

# FaceMesh indices; cheek names are landmark-side labels, not anatomical guarantees
# under mirrored footage. Both are always analysed identically.
FACE={'forehead':[10,109,67,103,54,68,104,69,108,151,337,299,333,298,284,332,297,338],
      'left_cheek':[50,101,205,206,207,187,123,116,117,118,119,120],
      'right_cheek':[280,330,425,426,427,411,352,345,346,347,348,349]}

class Tracker:
    def __init__(self,cfg):
        self.cfg=cfg
        self.face=mp.solutions.face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=False,
                     min_detection_confidence=.5,min_tracking_confidence=.5)
        self.hands=None
    def close(self):
        self.face.close()
        if self.hands is not None:self.hands.close()
    def extract(self,bgr,with_hands=True):
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); h,w=rgb.shape[:2]; result={}; shapes={}
        face=self.face.process(rgb)
        if face.multi_face_landmarks:
            pts=np.array([(p.x*w,p.y*h) for p in face.multi_face_landmarks[0].landmark])
            for name in self.cfg['face_rois']:
                self.add(rgb,name,pts[FACE[name]],result,shapes)
        if with_hands:
            if self.hands is None:self.hands=mp.solutions.hands.Hands(max_num_hands=2,min_detection_confidence=.5,min_tracking_confidence=.5)
            hands=self.hands.process(rgb)
            for i,lm in enumerate(hands.multi_hand_landmarks or []):
                side=hands.multi_handedness[i].classification[0].label
                pts=np.array([(p.x*w,p.y*h) for p in lm.landmark])
                span=np.linalg.norm(pts[5]-pts[17]); r=max(2,span*.13)
                centers={'palm_center':np.mean(pts[[0,5,9,13,17]],0),
                         'thenar':.5*pts[1]+.25*pts[2]+.25*pts[0],
                         'index_pad':.7*pts[8]+.3*pts[7]}
                for name in self.cfg['hand_rois']:
                    c=centers[name]; rr=r*(.55 if name=='index_pad' else 1)
                    self.add(rgb,f'{side}_{name}',c+np.array([[-rr,-rr],[rr,-rr],[rr,rr],[-rr,rr]]),result,shapes)
        return result,shapes
    def add(self,rgb,name,poly,result,shapes):
        h,w=rgb.shape[:2]; p=np.round(poly).astype(np.int32)
        if (p[:,0]<0).any() or (p[:,0]>=w).any() or (p[:,1]<0).any() or (p[:,1]>=h).any(): return
        mask=np.zeros((h,w),np.uint8); cv2.fillConvexPoly(mask,cv2.convexHull(p),255)
        x,y,bw,bh=cv2.boundingRect(p)
        if bw<8 or bh<8 or mask.sum()<255*40:return
        # Reject near-black, clipped/glare pixels and preserve their exclusion in patches.
        valid=(mask>0)&(rgb.min(2)>8)&(rgb.max(2)<245)
        if valid.sum()<.6*(mask>0).sum():return
        mean=rgb[valid].mean(0); crop=rgb[y:y+bh,x:x+bw].copy()
        crop[~valid[y:y+bh,x:x+bw]]=mean
        patch=cv2.resize(crop,(self.cfg['size'],self.cfg['size']),interpolation=cv2.INTER_AREA)
        result[name]=(patch,mean,np.array([x+bw/2,y+bh/2])/np.array([w,h]))
        shapes[name]=p
