import time

import torch
import torch.nn as nn

def get_net():
    net = nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Linear(256,128),
                        nn.ReLU(),nn.Linear(128,2))
    return net

x = torch.randn(size=(1,512))

class Benchmark:
    def __init__(self,description='done'):
        self.description = description

    def __enter__(self):
        self.timer = time.time()
    def __exit__(self,*args):
        pass
net = get_net()
with Benchmark('no torchscript'):
    for i in range(1000):
        net(x)
net = torch.jit.script(net)
with Benchmark('with torchscript'):
    for i in range(1000):
        net(x)


