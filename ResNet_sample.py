import torch
from torch import nn
from torch.nn import functional as F

#ResNet 将一层的输出直接加到后面某一层上,放在卷积BN之后，激活之前，使用1*1卷积减小通道数量
#残差块
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1_1conv=False, strides=1):
        super(Residual,self).__init__()
        self.conv1 = nn.Conv2d(input_channels,num_channels,kernel_size=3,stride=strides,padding=1)
        self.conv2 = nn.Conv2d(num_channels,num_channels,kernel_size=3,padding=1)
        if use_1_1conv:
            self.conv3 = nn.Conv2d(input_channels,num_channels,kernel_size=1,stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

blk = Residual(3,3)
X = torch.randn(4,3,8,8)
Y = blk(X)
print(Y.shape)

#增加通道，同时减半长和宽
blk1 = Residual(3,6,use_1_1conv=True,strides=2)
print(blk1(X).shape)

#图像预处理，第一层，长宽减半，通道变为64，在经过maxpool长宽再减半
b1 = nn.Sequential(nn.Conv2d(1,64,kernel_size=7,padding=3,stride=2),
                   nn.BatchNorm2d(64),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

#ResNet使用4个残差块,第一个块输入输出通道不变，以后每个块输出通道减半，长宽减半
#对第一个残差块做特殊处理
def resnet_block(input_channels, num_channels,num_residuals,first_block=False):
    blk = []
    for i in range(num_residuals):
        if i==0 and not first_block:
            blk.append(
                Residual(input_channels,num_channels,use_1_1conv=True,strides=2)
            )
        else:
            blk.append(Residual(num_channels,num_channels))
    return blk
#每个模块有两个残差块，一共四个模块
# origin: [1,1,224,224]
# b1(conv+pool): [1,64,112,112],[1,64,56,56]
# b2:[1,64,56,56]
# b3:[1,128,28,28]
# b4:[1,256,14,14]
# b5:[1,512,7,7]
# AdaptiveAvgPool2D:[1,512,1,1]
# flatten: [1,512]
# linear: [1,10]
b2 = nn.Sequential(*resnet_block(64,64,2,first_block=True))
b3 = nn.Sequential(*resnet_block(64,128,2))
b4 = nn.Sequential(*resnet_block(128,256,2))
b5 = nn.Sequential(*resnet_block(256,512,2))
#合并
#AdaptiveAvgPool2d自适应平均池化，参数为目标输出长和宽，channel数不变. 从【1，512，7，7】
#变为【1，512，1，1】
net = nn.Sequential(b1,b2,b3,b4,b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(),
                    nn.Linear(512,10))
X = torch.rand(size=(1,1,224,224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,' output shape:\t', X.shape)


