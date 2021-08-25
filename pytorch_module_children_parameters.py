#https://zhuanlan.zhihu.com/p/156127643
#https://zhuanlan.zhihu.com/p/349156416
from collections import OrderedDict

import torch
from torch import nn

cpu = torch.device('cpu')
gpu = torch.device('cuda:0')

linear = nn.Linear(2,3)
#to是一个in_place的操作，直接在原来tensor上进行修改，不会产生一个新的tensor
#torch.float32; torch.half
linear.to(device=gpu, dtype=torch.float64,non_blocking=True)
print(linear.weight)

net = nn.Linear(2,3)
#返回一个由tuple组成的list,每个tuple由(key,value)组成
net.state_dict()
OrderedDict([('weight', torch.tensor([[0.6516, 0.2560],
                                      [-0.4956, -0.2821],
                                      [-0.0977,  0.6979]])), ('bias', torch.tensor([-0.6155, 0.0552, 0.0171]))])
net.state_dict().keys()
odict_keys(['weight', 'bias'])

import torch
from torch import nn
net = torch.nn.Sequential(nn.Linear(2,2), nn.Linear(2,2))
net.state_dict()
# OrderedDict([('0.weight', tensor([[ 0.2749,  0.5589],
#         [-0.6178,  0.6300]])), ('0.bias', tensor([0.6257, 0.2103])), ('1.weight', tensor([[-0.2376, -0.2866],
#         [ 0.3227,  0.2447]])), ('1.bias', tensor([ 0.4327, -0.2617]))])
for name, param in net.parameters():
    print(name,param)