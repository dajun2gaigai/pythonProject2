import torch
from torch import nn

#卷积块
def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),
        nn.ReLU(),
        nn.Conv2d(input_channels,num_channels,kernel_size=3,padding=1)
    )

#Dense块
class DenseBlock(nn.Module):
    def __init__(self, num_convs,input_channels,num_channels):
        super(DenseBlock,self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(
                conv_block(num_channels*i + input_channels,num_channels,)
            )
        self.net = nn.Sequential(*layer)

    def forward(self,X):
        for blk in self.net:
            Y = blk(X)
            #连通DenseBlock中每个conv_block
            X = torch.cat((X,Y),dim=1)
        return X
#example
blk = DenseBlock(2,3,10)
X = torch.randn(3,3,8,8)
Y = blk(X)
print(Y.shape)

#过渡层，使用1*1卷积，减少通道数，控制模型参数数量
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),        nn.ReLU(),
        nn.Conv2d(input_channels,num_channels,kernel_size=1),
        nn.AvgPool2d(kernel_size=2,stride=2)
    )
blk = transition_block(23, 10)
Z = blk(Y)
print(Z.shape)

#DenseNet使用4个dense块，DenseNet通过过度层来减半长宽，以及通道数量
#num_channel为当前的通道数, growth_rate为dense块中每个conv块的输出维度，就是增长率
num_channels, growth_rate = 64, 32
blks = []
num_convs_in_dense_blocks = [4,4,4,4]
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    #通过enumerate逐层创建dense块
    blks.append(DenseBlock(num_convs,num_channels,growth_rate))
    #当前dense块运行完的维度
    num_channels += num_convs*growth_rate

#所有dense块创建完成后，增加过渡层减小长宽和通道数
    if i != len(num_convs_in_dense_blocks)-1:
        blks.append(transition_block(num_channels,num_channels//2))
        num_channels=num_channels//2

#最后连接上全局池化层和全连接层来输出结果
net = nn.Sequential(b1,*blks,
                    nn.BatchNorm2d(num_channels),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(),
                    nn.Linear(num_channels,10)
                    )

