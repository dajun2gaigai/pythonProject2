import torch
from torch import nn
from torch.nn import functional as F

b1 = nn.Sequential(nn.Conv2d(1,64,kernel_size=7,padding=3,stride=2),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3,padding=1,stride=2))
b2 = nn.Sequential(nn.Conv2d(64,64,kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64,192,kernel_size=3,padding=1),
                   )
#inception模块一般有四个线路组成，具体看下面
class Inception(nn.Module):
    def __init__(self,in_channels,c1,c2,c3,c4,**kwargs):
        super(Inception,self).__init__(**kwargs)
        #线路1，1*1，不改变长宽，改变通道数量
        self.p1_1 = nn.Conv2d(in_channels,c1,kernel_size=1)
        #线路2 先1*1卷积改变通道个数，接3*3padding的卷积
        self.p2_1 = nn.Conv2d(in_channels,c2[0],kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1)
        #线路3 先1*1卷积改变通道，接5*5padding
        self.p3_1 = nn.Conv2d(in_channels,c3[0],kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0],c3[1],kernel_size=5,padding=2)
        #线路4 先3*3padding池化,不改变长宽，不改变维度，接1*1卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3,padding=1,stride=1)
        self.p4_2 = nn.Conv2d(in_channels,c4,kernel_size=1)

    def forward(self,x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))

        return torch.cat((p1,p2,p3,p4),dim=1)
#每个Inception模块只改变channel数，不改变长宽，长宽由最后的maxpool改变
b3 = nn.Sequential(Inception(192,64,(96,128),(16,32),32),
                   Inception(256,128,(128,192),(32,96),64),
                   nn.MaxPool2d(kernel_size=3,padding=1,stride=2))
b4 = nn.Sequential(Inception(480,192,(96,208),(16,48),64),
                   Inception(512,160,(112,224),(24,64),64),
                   Inception(512,128,(128,256),(24,64),64),
                   Inception(512,112,(144,288),(32,64),64),
                   Inception(528,256,(160,320),(32,128),128),
                   nn.MaxPool2d(kernel_size=3,padding=1,stride=2))
#最后通过AdaptiveAvgPool和nn.Flatten()展开后，接linear实现分类
b5 = nn.Sequential(Inception(832,256,(160,320),(32,128),128),
                   Inception(832,384,(192,384),(48,128),128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())
net = nn.Sequential(b1,b2,b3,b4,b5,nn.Linear(1024,10))

X = torch.rand(size=(1,1,96,96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, ' output shape:\t', X.shape)

