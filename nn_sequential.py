from collections import OrderedDict
import torch
from torch import nn


model = nn.Sequential(
    nn.Conv2d(1,20,5),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(20,64,5),
    nn.ReLU(),
    nn.Linear(64,10)
)

model2 = nn.Sequential(
    OrderedDict(
        [('conv1',nn.Conv2d(1,20,5)),('relu',nn.ReLU()),('maxpool',nn.MaxPool2d(2)),
         ('conv2',nn.Conv2d(20,64,5)),('relu',nn.ReLU()),('linear',nn.Linear(64,10))]
    )
)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        seq = nn.Sequential()
        seq.add_module('conv1',nn.Conv2d(1,20,5))
        seq.add_module('relu',nn.ReLU())
        seq.add_module('maxpool',nn.MaxPool2d(2))

class Sequential(nn.Module):
    def __init__(self,*args,**kwargs):
        super.__init__(Sequential,self).__init__()
        if len(args)==1 and isinstance(args[0],OrderedDict):60
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            self.add_module(name, module)

    def fowward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

