import torch
from torch import nn
from d2l import torch as d2l
import matplotlib
from torch.nn import functional as F

scale = 0
W1 = torch.randn(size=(20,1,3,3))*scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50,20,5,5))*scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800,128))*scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128,10))*scale
b4 = torch.zeros(10)
params = [W1,b1,W2,b2,W3,b3,W4,b4]

def lenet(X,params):
    h1_conv = F.conv2d(input=X, weight=params[0],bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation,kernel_size=(2,2),stride=(2,2))
    h2_conv = F.conv2d(input=h1, weight=params[2],bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation,kernel_size=(2,2),stride=(2,2))
    h2 = h2.reshape(h2.shape[0],-1)
    h3_linear = torch.mm(h2,params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3,params[6]) + params[6]

    return y_hat

loss = nn.CrossEntropyLoss(reduction='none')

#将params拷贝到device
def get_params(params,device):
    new_params = [p.clone().to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params

#先报所有数据拷贝到data[0]所在的device相加，然后把sum在广播到所有device
def allreduce(data):
    for i in range(1,len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1,len(data)):
        data[i] = data[0].to(data[i].device)

#在定义数据时，直接制定data的device
print(torch.cuda.device_count())
data = [torch.ones((1,2),device=d2l.try_gpu(i))*(i+1) for i in range(2)]
print('before allreduce:\n',data[0],'\n',data[1])
allreduce(data)
print('after allreduce:\n',data[0],'\n',data[1])

#数据定以后分发到不同GPU，nn.parallel.scatter
data = torch.arange(20).reshape(4,5)
devices = [torch.device('cuda:0'),torch.device('cuda:1')]
#返回数据分割后的list，list中的每个元素在不同的GPU中运行
split = nn.parallel.scatter(data,devices)
print('input:\n',data)
print('load into\n',devices)
print('output\n',split)

#拆分数据到不同的GPU
def split_batch(X,y,devices):
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X,devices),nn.parallel.scatter(y,devices))
#定义train中的小批零训练 train_batch
def train_batch(X,y,device_params,devices,lr):
    X_shards, y_shards = split_batch(X,y,devices)
    #每个GPU上的运算
    ls = [
        loss(lenet(X_shard,device_W),y_shard).sum() for X_shard,y_shard,device_W in zip(X_shards,y_shards,device_params)
    ]

    for l in ls:
        l.backward()
    #把每个GPU上的梯度相加，广播到所有GPU
    with torch.no_grad():
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad for c in range(len(devices))])
    #每个GPU上更新参数模型，使用全尺寸mini_batch
    for param in device_params:
        d2l.sgd(param,lr,X.shape[0])
'''
1 初始化网络参数
2 复制网路到每个GPU
3 将mini-batch数据分割后分配给每个GPU
4 各个GPU计算损失和梯度
5 聚合梯度，广播，梯度下降更新参数'''
#整体训练过程，分配GPU，将所有模型参数复制到设备，每个小批量使用train_batch运用多个GPU并行处理
def train(num_gpus, batch_size,lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    #复制参数到每个GPU
    device_params = [get_params(params,d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch','test acc', xlim=[1,num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X,y in train_iter:
            #将每个mini-batch中的数据分割后，分配给每个GPU
            train_batch(X,y,device_params,devices,lr)
            torch.cuda.synchronize()
        timer.stop()
        animator.add(epoch+1,(d2l.evaluate_accuracy_gpu(lambda x: lenet(x,device_params[0]),test_iter,devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
