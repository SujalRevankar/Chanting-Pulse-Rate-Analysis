import torch
from torch import nn

class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.AvgPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.AvgPool2d(2),
            nn.Conv2d(32,32,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1))
        self.lstm=nn.LSTM(32,64,batch_first=True)
        self.output=nn.Linear(64,1)
    def forward(self,x):
        # [batch,time,height,width,RGB] -> one waveform sample per frame
        b,t,h,w,c=x.shape
        z=self.cnn(x.reshape(b*t,h,w,c).permute(0,3,1,2)).flatten(1).reshape(b,t,32)
        z,_=self.lstm(z)
        return self.output(z).squeeze(-1)

def correlation_loss(pred,target):
    p=pred-pred.mean(1,keepdim=True); y=target-target.mean(1,keepdim=True)
    r=(p*y).sum(1)/((p.square().sum(1)*y.square().sum(1)).sqrt()+1e-8)
    return 1-r.mean()
