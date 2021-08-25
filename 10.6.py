import argparse
import torch
import torchvision.datasets
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F

'''
PyTorch的计算将尝试使用所有CPU核心
gpu设备只代表一个卡和相应的显存。如果有多个GPU，我们使用torch.cuda.device(f'cuda:{i}')来表示第 𝑖 块GPU（ 𝑖 从0开始）。
另外，gpu:0和gpu是等价的
打印张量或将张量转换为NumPy格式时。如果数据不在内存中，框架会首先将其复制到内存中，这会导致额外的传输开销
'''
torch.device('cpu')
torch.device('cuda')
# torch.cuda.device(f'cuda:{i}')
torch.cuda.device_count()
print(torch.cuda.device_count())

def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else torch.device('cpu')

#默认在cpu
x = torch.tensor([1,2,3])
x.device
x = torch.tensor([1,2,3], device=try_gpu())
x = torch.tensor([1,2,3],device=torch.device('cuda:0'))
print(x)

#不在同一个GPU上的数据不能执行相加操作，需要复制到一起
X = torch.ones((2,3), device=torch.device('cuda:0'))
Y = torch.ones((2,3), device=torch.device('cpu'))
Z = X + Y.clone().to(X.device)
Z = X + Y.cuda(0)
print(Z)

#神经网络参数查看
net = nn.Sequential(nn.Linear(3,10))
net = net.to(try_gpu())
net(X)
print(net[0])
print(net[0].weight.data.device)

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(Residual,self).__init__()
        self.conv1 = nn.Conv2d(input_channels,num_channels,kernel_size=3,stride=strides,padding=1)
        self.conv2 = nn.Conv2d(num_channels,num_channels,kernel_size=3,padding=1)
        if use_1x1conv:
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

def resnet18(num_classes, in_channels=1):
    """稍加修改的ResNet-18模型。"""
    # ResNet使用4个残差块,第一个块输入输出通道不变，以后每个块输出通道减半，长宽减半
    # 对第一个残差块做特殊处理,不改变长宽及通道
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(
                    Residual(in_channels, out_channels, use_1x1conv=True,
                                 strides=2))
            else:
                blk.append(Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # 该模型使用了更小的卷积核、步长和填充，且删除了最大池化层。
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64), nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
    net.add_module("fc",
                   nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))
    return net
def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

# Defined in file: ./chapter_deep-learning-computation/use-gpu.md
def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

net = resnet18(10)
devices = try_all_gpus()

def train(net,num_gpus,batch_size,lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    '''
    1 初始化网络参数
    2 复制网路到每个GPU
    3 将mini-batch数据分割后分配给每个GPU
    4 各个GPU计算损失和梯度
    5 聚合梯度，广播，梯度下降更新参数'''
    def init_weights(m):
        if type(m) in [nn.Linear,nn.Conv2d]:
            nn.init.normal_(m.weight,std=0.01)
    net.apply(init_weights)

    #在多GPU上设置模型
    net = nn.DataParallel(net,device_ids=devices)
    optimizer = torch.optim.SGD(net.parameters(),lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(),10
    animator = d2l.Animator('epoch','test acc', xlim=[1,num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X,y in train_iter:
            optimizer.zero_grad()
            X,y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X),y)
            l.backward()
            optimizer.step()
        timer.stop()
        animator.add(epoch+1,(d2l.evaluate_accuracy_gpu(net,test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f},{timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
train(net,num_gpus=1,batch_size=1,lr=0.1)