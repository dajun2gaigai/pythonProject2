#https://zhuanlan.zhihu.com/p/156127643
#https://zhuanlan.zhihu.com/p/349156416

from collections import OrderedDict

'''
state_dict返回的OrderedDict, OrderedDict是一个由tuple(key,value)组成的list, list的每个元素是由字典的key和value组成的一个tuple，也相当于一个字典了
其他的都是返回一个generator
module, children, parameters都是一个list
named_module, named_children, named_parameters都是由tuple(key,value)组成的一个list
'''
import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, num_class=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3),
            nn.BatchNorm2d(9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(9 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_class)
        )

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output


model = Net()

#modules, named_modules,children, named_children, parameters, named_parameters, state_dict
#state_dict返回的是字典，其他返回的都是一个generator
module = model.modules()
named_module = model.named_modules()
children = model.children()
named_childre = model.named_children()
params = model.parameters()
named_params = model.named_parameters()
state_dict = model.state_dict()

#取出generator中的各个元素
model_modules = [x for x in model.modules()]
model_named_modules = [x for x in model.named_modules()]
model_children = [x for x in model.children()]
model_named_children = [x for x in model.named_children()]
model_parameters = [x for x in model.parameters()]
model_named_parameters = [x for x in model.named_parameters()]

# print(model_modules)
# [Net(
#   (features): Sequential(
#     (0): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
#     (1): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU(inplace=True)
#     (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (4): Conv2d(6, 9, kernel_size=(3, 3), stride=(1, 1))
#     (5): BatchNorm2d(9, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (6): ReLU(inplace=True)
#     (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (classifier): Sequential(
#     (0): Linear(in_features=576, out_features=128, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=128, out_features=10, bias=True)
#   )
# ),
# Sequential(
#   (0): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
#   (1): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU(inplace=True)
#   (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (4): Conv2d(6, 9, kernel_size=(3, 3), stride=(1, 1))
#   (5): BatchNorm2d(9, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (6): ReLU(inplace=True)
#   (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# ),
# Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1)),
# BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
# ReLU(inplace=True),
# MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
# Conv2d(6, 9, kernel_size=(3, 3), stride=(1, 1)),
# BatchNorm2d(9, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
# ReLU(inplace=True),
# MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
# Sequential(
#   (0): Linear(in_features=576, out_features=128, bias=True)
#   (1): ReLU(inplace=True)
#   (2): Dropout(p=0.5, inplace=False)
#   (3): Linear(in_features=128, out_features=10, bias=True)
# ),
# Linear(in_features=576, out_features=128, bias=True),
# ReLU(inplace=True),
# Dropout(p=0.5, inplace=False),
# Linear(in_features=128, out_features=10, bias=True)]



# print(model_named_modules)

# [('', Net(
#   (features): Sequential(
#     (0): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
#     (1): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (2): ReLU(inplace=True)
#     (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (4): Conv2d(6, 9, kernel_size=(3, 3), stride=(1, 1))
#     (5): BatchNorm2d(9, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (6): ReLU(inplace=True)
#     (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (classifier): Sequential(
#     (0): Linear(in_features=576, out_features=128, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=128, out_features=10, bias=True)
#   )
# )),
# ('features', Sequential(
#   (0): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
#   (1): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU(inplace=True)
#   (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (4): Conv2d(6, 9, kernel_size=(3, 3), stride=(1, 1))
#   (5): BatchNorm2d(9, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (6): ReLU(inplace=True)
#   (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# )),
# ('features.0', Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))), ('features.1', BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)), ('features.2', ReLU(inplace=True)), ('features.3', MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)), ('features.4', Conv2d(6, 9, kernel_size=(3, 3), stride=(1, 1))), ('features.5', BatchNorm2d(9, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)), ('features.6', ReLU(inplace=True)), ('features.7', MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)),
# ('classifier', Sequential(
#   (0): Linear(in_features=576, out_features=128, bias=True)
#   (1): ReLU(inplace=True)
#   (2): Dropout(p=0.5, inplace=False)
#   (3): Linear(in_features=128, out_features=10, bias=True)
# )), ('classifier.0', Linear(in_features=576, out_features=128, bias=True)), ('classifier.1', ReLU(inplace=True)), ('classifier.2', Dropout(p=0.5, inplace=False)), ('classifier.3', Linear(in_features=128, out_features=10, bias=True))]

# print(model_children)
# [Sequential(
#   (0): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
#   (1): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU(inplace=True)
#   (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (4): Conv2d(6, 9, kernel_size=(3, 3), stride=(1, 1))
#   (5): BatchNorm2d(9, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (6): ReLU(inplace=True)
#   (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# ),
# Sequential(
#   (0): Linear(in_features=576, out_features=128, bias=True)
#   (1): ReLU(inplace=True)
#   (2): Dropout(p=0.5, inplace=False)
#   (3): Linear(in_features=128, out_features=10, bias=True)
# )]

# print(model_named_children)
# [('features', Sequential(
#   (0): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
#   (1): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU(inplace=True)
#   (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (4): Conv2d(6, 9, kernel_size=(3, 3), stride=(1, 1))
#   (5): BatchNorm2d(9, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (6): ReLU(inplace=True)
#   (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
# )),
# ('classifier', Sequential(
#   (0): Linear(in_features=576, out_features=128, bias=True)
#   (1): ReLU(inplace=True)
#   (2): Dropout(p=0.5, inplace=False)
#   (3): Linear(in_features=128, out_features=10, bias=True)
# ))]



# print(model_parameters)
# [Parameter containing:
#  tensor([[[[ 0.1200, -0.1627, -0.0841],
#            [-0.1369, -0.1525,  0.0541],
#            [ 0.1203,  0.0564,  0.0908]],
#            ……
#           [[-0.1587,  0.0735, -0.0066],
#            [ 0.0210,  0.0257, -0.0838],
#            [-0.1797,  0.0675,  0.1282]]]], requires_grad=True),
#  Parameter containing:
#  tensor([-0.1251,  0.1673,  0.1241, -0.1876,  0.0683,  0.0346],
#         requires_grad=True),
#  Parameter containing:
#  tensor([0.0072, 0.0272, 0.8620, 0.0633, 0.9411, 0.2971], requires_grad=True),
#  Parameter containing:
#  tensor([0., 0., 0., 0., 0., 0.], requires_grad=True),
#  Parameter containing:
#  tensor([[[[ 0.0632, -0.1078, -0.0800],
#            [-0.0488,  0.0167,  0.0473],
#            [-0.0743,  0.0469, -0.1214]],
#            ……
#           [[-0.1067, -0.0851,  0.0498],
#            [-0.0695,  0.0380, -0.0289],
#            [-0.0700,  0.0969, -0.0557]]]], requires_grad=True),
#  Parameter containing:
#  tensor([-0.0608,  0.0154,  0.0231,  0.0886, -0.0577,  0.0658, -0.1135, -0.0221,
#           0.0991], requires_grad=True),
#  Parameter containing:
#  tensor([0.2514, 0.1924, 0.9139, 0.8075, 0.6851, 0.4522, 0.5963, 0.8135, 0.4010],
#         requires_grad=True),
#  Parameter containing:
#  tensor([0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=True),
#  Parameter containing:
#  tensor([[ 0.0223,  0.0079, -0.0332,  ..., -0.0394,  0.0291,  0.0068],
#          [ 0.0037, -0.0079,  0.0011,  ..., -0.0277, -0.0273,  0.0009],
#          [ 0.0150, -0.0110,  0.0319,  ..., -0.0110, -0.0072, -0.0333],
#          ...,
#          [-0.0274, -0.0296, -0.0156,  ...,  0.0359, -0.0303, -0.0114],
#          [ 0.0222,  0.0243, -0.0115,  ...,  0.0369, -0.0347,  0.0291],
#          [ 0.0045,  0.0156,  0.0281,  ..., -0.0348, -0.0370, -0.0152]],
#         requires_grad=True),
#  Parameter containing:
#  tensor([ 0.0072, -0.0399, -0.0138,  0.0062, -0.0099, -0.0006, -0.0142, -0.0337,
#           ……
#          -0.0370, -0.0121, -0.0348, -0.0200, -0.0285,  0.0367,  0.0050, -0.0166],
#         requires_grad=True),
#  Parameter containing:
#  tensor([[-0.0130,  0.0301,  0.0721,  ..., -0.0634,  0.0325, -0.0830],
#          [-0.0086, -0.0374, -0.0281,  ..., -0.0543,  0.0105,  0.0822],
#          [-0.0305,  0.0047, -0.0090,  ...,  0.0370, -0.0187,  0.0824],
#          ...,
#          [ 0.0529, -0.0236,  0.0219,  ...,  0.0250,  0.0620, -0.0446],
#          [ 0.0077, -0.0576,  0.0600,  ..., -0.0412, -0.0290,  0.0103],
#          [ 0.0375, -0.0147,  0.0622,  ...,  0.0350,  0.0179,  0.0667]],
#         requires_grad=True),
#  Parameter containing:
#  tensor([-0.0709, -0.0675, -0.0492,  0.0694,  0.0390, -0.0861, -0.0427, -0.0638,
#          -0.0123,  0.0845], requires_grad=True)]


# print(model_named_parameters)
# [('feature.0.weight', torch.Size(6,3,3,3)),
#  ('feature.0.bias',torch.Size(6))]

# t = torch.tensor([[[[-0.0335, 0.0487, 0.1817],
#                     [-0.0631,  0.0480, -0.1731],
#                     [ 0.0836,  0.0312,  0.1136]],
#
#                    [[ 0.1846, -0.1078, -0.1310],
#           [-0.0200,  0.0734,  0.1245],
#           [ 0.0039,  0.0888,  0.1187]],
#
#                    [[-0.0379,  0.0046,  0.0163],
#           [-0.0284,  0.0276,  0.1917],
#           [-0.0856, -0.0012,  0.0106]]],
#
#
#                   [[[-0.1604,  0.1623, -0.0529],
#           [ 0.0166,  0.1293, -0.0287],
#           [ 0.1453, -0.0179, -0.1514]],
#
#          [[ 0.0277, -0.0244, -0.1455],
#           [-0.0497, -0.1653, -0.0341],
#           [-0.0815,  0.0623, -0.0715]],
#
#          [[-0.0577,  0.0582, -0.0463],
#           [-0.0424, -0.1438,  0.1028],
#           [-0.0512, -0.0847, -0.0896]]],
#
#
#                   [[[ 0.1736, -0.0407, -0.0895],
#           [ 0.1410,  0.0788, -0.0344],
#           [-0.0038, -0.1790, -0.0734]],
#
#          [[ 0.0271, -0.0300,  0.0809],
#           [ 0.1639, -0.1700,  0.0024],
#           [ 0.0648,  0.0465,  0.0350]],
#
#          [[-0.1214,  0.0069, -0.1067],
#           [ 0.0192, -0.0298,  0.0514],
#           [ 0.0845, -0.1134,  0.0706]]],
#
#
#                   [[[-0.0281,  0.0702,  0.0040],
#           [-0.1835, -0.0252,  0.1627],
#           [ 0.1529, -0.1165, -0.0760]],
#
#          [[ 0.1204, -0.1693, -0.0409],
#           [-0.0656, -0.0067, -0.1475],
#           [ 0.0853, -0.0608, -0.1894]],
#
#          [[-0.0830,  0.0361,  0.1325],
#           [-0.1404, -0.0961,  0.1586],
#           [ 0.1776, -0.0284, -0.1033]]],
#
#
#                   [[[-0.0615,  0.1388,  0.1543],
#           [ 0.0871,  0.1792,  0.1721],
#           [ 0.0752, -0.0684, -0.0876]],
#
#          [[ 0.0368,  0.0081, -0.0111],
#           [-0.0451, -0.1235,  0.1505],
#           [-0.0403, -0.0218,  0.0468]],
#
#          [[ 0.1906,  0.0157, -0.1608],
#           [ 0.1854, -0.0196, -0.1827],
#           [-0.0101, -0.0547,  0.0958]]],
#
#
#                   [[[-0.0743, -0.1639, -0.1787],
#           [-0.1850, -0.0553,  0.0422],
#           [ 0.0310, -0.1477,  0.0789]],
#
#          [[-0.0599,  0.1650, -0.1278],
#           [ 0.1325,  0.1752, -0.1721],
#           [-0.1762, -0.0851,  0.0304]],
#
#          [[ 0.0246,  0.0713,  0.0113],
#           [ 0.1506, -0.0265,  0.0967],
#           [ 0.0266, -0.0688, -0.1239]]]], requires_grad=True)
# print(t.size())

# [('features.0.weight', Parameter containing:
#   tensor([[[[ 0.1200, -0.1627, -0.0841],
#             [-0.1369, -0.1525,  0.0541],
#             [ 0.1203,  0.0564,  0.0908]],
#            ……
#            [[-0.1587,  0.0735, -0.0066],
#             [ 0.0210,  0.0257, -0.0838],
#             [-0.1797,  0.0675,  0.1282]]]], requires_grad=True)),
#  ('features.0.bias', Parameter containing:
#   tensor([-0.1251,  0.1673,  0.1241, -0.1876,  0.0683,  0.0346],
#          requires_grad=True)),
#  ('features.1.weight', Parameter containing:
#   tensor([0.0072, 0.0272, 0.8620, 0.0633, 0.9411, 0.2971], requires_grad=True)),
#  ('features.1.bias', Parameter containing:
#   tensor([0., 0., 0., 0., 0., 0.], requires_grad=True)),
#  ('features.4.weight', Parameter containing:
#   tensor([[[[ 0.0632, -0.1078, -0.0800],
#             [-0.0488,  0.0167,  0.0473],
#             [-0.0743,  0.0469, -0.1214]],
#            ……
#            [[-0.1067, -0.0851,  0.0498],
#             [-0.0695,  0.0380, -0.0289],
#             [-0.0700,  0.0969, -0.0557]]]], requires_grad=True)),
#  ('features.4.bias', Parameter containing:
#   tensor([-0.0608,  0.0154,  0.0231,  0.0886, -0.0577,  0.0658, -0.1135, -0.0221,
#            0.0991], requires_grad=True)),
#  ('features.5.weight', Parameter containing:
#   tensor([0.2514, 0.1924, 0.9139, 0.8075, 0.6851, 0.4522, 0.5963, 0.8135, 0.4010],
#          requires_grad=True)),
#  ('features.5.bias', Parameter containing:
#   tensor([0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=True)),
#  ('classifier.0.weight', Parameter containing:
#   tensor([[ 0.0223,  0.0079, -0.0332,  ..., -0.0394,  0.0291,  0.0068],
#           ……
#           [ 0.0045,  0.0156,  0.0281,  ..., -0.0348, -0.0370, -0.0152]],
#          requires_grad=True)),
#  ('classifier.0.bias', Parameter containing:
#   tensor([ 0.0072, -0.0399, -0.0138,  0.0062, -0.0099, -0.0006, -0.0142, -0.0337,
#            ……
#           -0.0370, -0.0121, -0.0348, -0.0200, -0.0285,  0.0367,  0.0050, -0.0166],
#          requires_grad=True)),
#  ('classifier.3.weight', Parameter containing:
#   tensor([[-0.0130,  0.0301,  0.0721,  ..., -0.0634,  0.0325, -0.0830],
#           [-0.0086, -0.0374, -0.0281,  ..., -0.0543,  0.0105,  0.0822],
#           [-0.0305,  0.0047, -0.0090,  ...,  0.0370, -0.0187,  0.0824],
#           ...,
#           [ 0.0529, -0.0236,  0.0219,  ...,  0.0250,  0.0620, -0.0446],
#           [ 0.0077, -0.0576,  0.0600,  ..., -0.0412, -0.0290,  0.0103],
#           [ 0.0375, -0.0147,  0.0622,  ...,  0.0350,  0.0179,  0.0667]],
#          requires_grad=True)),
#  ('classifier.3.bias', Parameter containing:
#   tensor([-0.0709, -0.0675, -0.0492,  0.0694,  0.0390, -0.0861, -0.0427, -0.0638,
#           -0.0123,  0.0845], requires_grad=True))]

def children(m: nn.Module):
    return list(m.children())


def num_children(m: nn.Module) -> int:
    return len(children(m))

# print(num_children(model))
# print(children(model))

flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

# [Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1)),
#  BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#  ReLU(inplace=True),
#  MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
#  Conv2d(6, 9, kernel_size=(3, 3), stride=(1, 1)),
#  BatchNorm2d(9, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#  ReLU(inplace=True),
#  MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
#  Linear(in_features=576, out_features=128, bias=True),
#  ReLU(inplace=True),
#  Dropout(p=0.5, inplace=False),
#  Linear(in_features=128, out_features=10, bias=True)]
#返回的是一个由最小的nn.Module子类构成的list
print(flatten_model(model))
# [Sequential(
#   (0): Conv2d(3, 6, kernel_size=(3, 3), stride=(1, 1))
#   (1): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (2): ReLU(inplace=True)
#   (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (4): Conv2d(6, 9, kernel_size=(3, 3), stride=(1, 1))
#   (5): BatchNorm2d(9, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#   (6): ReLU(inplace=True)
#   (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   (8): Linear(in_features=576, out_features=128, bias=True)
#   (9): ReLU(inplace=True)
#   (10): Dropout(p=0.5, inplace=False)
#   (11): Linear(in_features=128, out_features=10, bias=True)
# )]
print(get_layer_groups(model))