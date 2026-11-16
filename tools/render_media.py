"""Render fictional ROI illustration and silent phase demo from supplied face CSVs.
Run from any directory: python tools/render_media.py
Needs numpy, pandas, matplotlib, Pillow and ffmpeg on PATH.
"""
from pathlib import Path
import math, subprocess
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]; A=P/'assets';R=P/'results/face_example'
W,H=1200,720; FPS=12;SECONDS=24
COLORS={'forehead':'#13a89e','left_cheek':'#7c61bc','right_cheek':'#e58337'}
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
if not Path(FONT).exists():
 FONT=str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf')
def font(n):return ImageFont.truetype(FONT,n)
def txt(d,xy,s,n=22,c='#172b41'):d.text(xy,s,font=font(n),fill=c)
sheet=Image.open(A/'fictional-character-sheet.png').convert('RGB')
portraits=[sheet.crop((i*sheet.width//2,0,(i+1)*sheet.width//2,sheet.height)).resize((384,512)) for i in range(2)]
# Approximate overlays located manually on fictional artwork, NOT detected landmarks.
polys={'forehead':[(139,111),(241,111),(256,151),(129,151)],'left_cheek':[(113,206),(145,213),(158,242),(121,258)],'right_cheek':[(242,213),(273,206),(267,258),(229,242)]}
def portrait(canvas,xy,open_mouth=False):
 canvas.paste(portraits[int(open_mouth)],xy);layer=Image.new('RGBA',canvas.size);d=ImageDraw.Draw(layer)
 for name,poly in polys.items():
  c=COLORS[name];rgb=tuple(int(c[i:i+2],16) for i in (1,3,5));pts=[(x+xy[0],y+xy[1]) for x,y in poly];d.polygon(pts,fill=(*rgb,55));d.line(pts+[pts[0]],fill=(*rgb,255),width=3)
 canvas.paste(Image.alpha_composite(canvas.convert('RGBA'),layer).convert('RGB'))
# Static ROI plate.
im=Image.new('RGB',(1200,720),'#f4f7fa');d=ImageDraw.Draw(im)
txt(d,(45,30),'Facial regions used by the code',34);txt(d,(45,82),'Schematic locations on an AI-generated fictional person',20,'#52667a')
portrait(im,(85,145));d=ImageDraw.Draw(im)
for i,(name,col) in enumerate(COLORS.items()):
 y=180+i*115;d.rectangle((540,y+4,552,y+42),fill=col);txt(d,(578,y),name.replace('_',' ').title(),27,col)
 txt(d,(578,y+43),'Convex hull of selected Face Mesh landmarks',18)
txt(d,(45,665),'Not participant footage or a measured tracking result. Exact indices: docs/METHODS.md',18)
im.save(A/'face-rois.png')
# Scientific figures from exact exports.
s=pd.read_csv(R/'segments.csv'); win=pd.read_csv(R/'windows.csv')
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(11,5.5))
for roi,col in COLORS.items():
 g=s[s.roi==roi].sort_values('segment');ax.plot(g.segment,g.average_bpm,'o-',color=col,label=roi.replace('_',' '))
ax.set_xticks(range(4),['Silence 1','Ha','Silence 2','Ya']);ax.set_ylabel('Duration-weighted estimated pulse rate (BPM)');ax.set_ylim(70,100);ax.legend();ax.grid(axis='y',alpha=.2);ax.set_title('One supplied face output set · accepted windows only')
fig.text(.1,.015,'Variation is observational; chanting recordings have no contact reference.',fontsize=10);fig.tight_layout(rect=(0,.04,1,1));fig.savefig(A/'face-phase-comparison.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(11,5.5))
for roi,col in COLORS.items():
 g=win[win.roi==roi].sort_values('start_s');y=g.model_bpm.where(g.status=='accepted');ax.plot((g.start_s+g.end_s)/2,y,'o-',color=col,label=roi.replace('_',' '))
ax.set(xlim=(0,240),ylim=(70,100),xlabel='Recording time (s)',ylabel='Estimated pulse rate (BPM)',title='Face windows · rejected windows break the plotted series');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(A/'face-window-timeline.png',dpi=160);plt.close(fig)
labels=['Silence 1','Ha','Silence 2','Ya']
means={roi:s[s.roi==roi].sort_values('segment').average_bpm.to_numpy() for roi in COLORS}
coverage={roi:s[s.roi==roi].sort_values('segment').coverage.to_numpy() for roi in COLORS}
def frame(k):
 phase=min(3,k//(FPS*6));local=k%(FPS*6);chant=phase in (1,3)
 im=Image.new('RGB',(W,H),'#f4f7fa');d=ImageDraw.Draw(im)
 d.rectangle((0,0,W,110),fill='#112d40');txt(d,(34,18),'Face-based pulse-rate analysis during chanting',30,'white');txt(d,(35,65),'ILLUSTRATIVE ANIMATION  /  Fictional person + supplied numerical estimates',17,'#bcd5df')
 portrait(im,(25,120),chant and (local//4)%2==0);d=ImageDraw.Draw(im)
 txt(d,(48,635),'Schematic face regions',22);txt(d,(48,665),'No participant footage or chant audio',16,'#52667a')
 txt(d,(455,128),f'{phase+1:02d}  /  {labels[phase]}',32)
 txt(d,(455,177),f'Recording phase: {phase*60}–{(phase+1)*60} s  |  shown in 6 s',20,'#52667a')
 # Discrete phase means, never a fabricated instantaneous waveform.
 x0,x1,y0,y1=510,1115,270,470
 for tick in [70,80,90,100]:
  y=y1-(tick-70)/30*(y1-y0);d.line((x0,y,x1,y),fill='#dce5eb',width=1);txt(d,(458,y-10),str(tick),16)
 for j in range(4):txt(d,(x0+j*(x1-x0)/3-25,486),['S1','Ha','S2','Ya'][j],18)
 for roi,col in COLORS.items():
  pts=[]
  for j in range(phase+1):
   x=x0+j*(x1-x0)/3;y=y1-(means[roi][j]-70)/30*(y1-y0);pts.append((x,y))
  if len(pts)>1:d.line(pts,fill=col,width=3)
  for x,y in pts:d.ellipse((x-5,y-5,x+5,y+5),fill=col)
 txt(d,(455,222),'PHASE AVERAGES  ·  BPM',18)
 for i,(roi,col) in enumerate(COLORS.items()):
  x=455+i*239;txt(d,(x,540),roi.replace('_',' ').title(),19,col);txt(d,(x,570),f'{means[roi][phase]:.2f} BPM',26,col);txt(d,(x,609),f'Accepted coverage: {coverage[roi][phase]:.0%}',15)
 txt(d,(455,660),'Observed variation does not establish a chanting effect.',19)
 return im
if __name__=='__main__':
 proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(A/'chanting-demo.mp4')],stdin=subprocess.PIPE)
 gif=[]
 for k in range(FPS*SECONDS):
  im=frame(k);proc.stdin.write(im.tobytes())
  if k%2==0:gif.append(im.resize((900,540)).quantize(colors=128))
 proc.stdin.close();assert proc.wait()==0
 gif[0].save(A/'chanting-demo.gif',save_all=True,append_images=gif[1:],duration=[170 if i%3!=2 else 160 for i in range(len(gif))],loop=0,optimize=False)
 frame(FPS*8).save(A/'demo-preview.png')
 print('Rendered 24-second MP4 and GIF, ROI plate and two figures.')
