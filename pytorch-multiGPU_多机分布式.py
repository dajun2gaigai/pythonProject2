#https://www.zhihu.com/column/c_1276098562691452928
#https://zhuanlan.zhihu.com/p/178402798

import torch.nn.parallel.DataParallel
import torch.nn.parallel.DistributedDataParallel

from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

#总
python torch.cuda.set_device(i)
'''构建模型：其中，i 应该为 0 到 N - 1 之间。'''
torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
train_set = torchvision.datasets.CIFAR10()
train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=122,sampler=train_sampler)
model = DistributedDataParallel(model, device_ids=[i], output_device=i])


#torch.distributed.launch: 可以单节点多GPU(单节点多进程），或者多节点多GPU（多节点多进程）
#env:// execute script
#单节点多GPU: 一个机器上有多个GPU`
python -m torch.distributed.launch --nproc_per_node=3 mypython.py(*args,**kwargs)
#多节点多GPU：多个机器，每个机器上有多个GPU
#node 0
python -m torch.distributed.launch --nproc_per_node=3 nnodes=2 node_rank=0 --master_addr='127.0.0.1'
     --master_port=1234 mypython.py(*args,**kwargs))
#node 1
python -m torch.distributed.launch --nproc_per_node=3 nnodes=2 node_rank=1 --master_addr='...'
    --master_port 1234 mypython.py(*args,**kwargs))

#1 env:// execute script
python -m torch.distributed.launch --nproc_per_node=3 --nnode=2 --node_rank=0 --master_addr='127.0.0.1' --master_port=123456 env_init.py
#2 TCP: execute TCP script
python tcp_init.py --init_method tcp://192.168.0.1:12334 --rank 1 --world_size 2

#3 file:// execute script
python file_init.py --init_method file://mnt/nfs/sharedfile --rank 0 --world_size 2


#TCP_init.py
#TCP TCP初始化
parser = argparse.ArgumentParser(description='Pytorch distributed training on CIFAR-10')
parser.add_argument('--rank',default=0, help='rank of current process')
parser.add_argument('--world_size',default=2, help='world size')
parser.add_argument('--init_method',default='tcp://127.0.0.1:23456',help='init_method')
args = parser.parse_args()

net = Net()
net = net.cuda()
net = torch.nn.parallel.DistributedDataParallel(net)

#每个进程负责多块GPU, 实现GPU模型并行
class ToyModel(nn.module):
    def init(self, dev0,dev1):
        super(ToyModel,self).__init__()
        self.dev0 = dev0
        self.dev1 = dev1
        self.net1 = torch.nn.Linear(10,10).to(dev0)
        self.relu = nn.ReLU()
        self.net2 = torch.nn.Linar(10,10).to(dev1)
    def forward(self,x):
        x = x.to(self.dev0)
        x = self.relu(self.net1(x))
        x = self.to(self.dev1)
        x = self.net2(x)
'''
dev0=rank*2
dev1=rank*2+1
mp_model=ToyModel(dev0,dev1)
ddp_mp_model=DDP(mp_model)
'''

python -m torch.distributed.launch --nproc_per_node=3 --nnodes=2 --rank=0 --master_addr='127.0.0.1' --master_port=1234 env.py --local_rank=[0,1,2]
#env初始化 env_init.py 为每个node配置本机进程以及每个进程对应的GPU
import torch.distributed as dist
import torch.utils.data.distributed
import argparse
parser = argparse.ArgumentParser()
#注意这个参数，必须要以这种形式指定，即使代码中不使用。因为 launch 工具默认传递该参数,
#when we use model parallel, the --local_rank should be a list
parser.add_argument('--local_rank',type=int)
args = parser.parse_args()
dist.init_process_group(backend='nccl',init_method='env://')
train_set = torchvision.datasets.CIFAR10(root='../data',
                                         train=True,
                                         download=download,
                                         transform=transform)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
train_loader = torch.utils.data.dataloader(train_set,batch_size=batch_size,sampler=train_sampler)
#根据local_rank,配置当前进程使用的GPU
net = Net()
device = torch.device('cuda',args.local_rank)
net = net.to(device)
net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[args.local_rank],output_device=args.local_rank)
#scrip执行方式, 只在rank0中填入参数就行
python -m torch.distributed.launch --nproc_per_node=3 --nnode=2 --node_rank=0 --master_addr='127.0.0.1' --master_port=123456 env_init.py

#初始化node进程组 init_process_group
#init_method默认使用env.py中的init_method
#timeout设置为30分钟
#
group = torch.distributed.init_process_group(backend='nccl',init_mehtod=None,
                                     timeout=datetime.timedelta(0,1800),
                                     world_size=-1,
                                     rank=-1,
                                     store=None)
#new_group:所有进程的任意子集来创建新组。其返回一个分组句柄，可作为 collectives 相关函数的 group 参数 。
# collectives 是分布式函数，用于特定编程模式中的信息交换
#ranks：指定新分组内的成员的 ranks 列表,参数为一个列表list ，其中每个元素为 int 型
#返回new_group这个进程的信息
new_group = torch.distributed.new_group(ranks=None,
                            timeout=datetime.timedelta(0,1800),
                            backend=None)
#get the attributes of the group
torch.distributed.get_backend(new_group)
torch.distributed.get_rank(new_group)
#返回这个进程组内的进程数量,什么也不填返回job的进程数
torch.distributed.get_world_size(new_group)
#检测
torch.distributed.is_initialized()
torch.distributed.is_mpi_available()
torch.distributed.is_nccl_available()
torch.distributed.is_gloo_available()

#1 实例mpidist.py, when the backed is MPI
import torch
import torch.distributed as dist

def main(rank,world):
    if rank == 0:
        x = torch.Tensor([1.,-1.])
        dist.send(x,dst=1)
        print('x has been sent to rank1 process')
    else:
        z = torch.Tensor([0.,0.])
        dist.recv(z,src=0)
        print('current rank received data from rank0')
if __name__ == '__main__':
    dist.init_process_group(backend='mpi')
    main(dist.get_rank(),dist.get_world_size())
#执行MPI
mpiexec -n 2 -ppn 1 -hosts miriad2a miriad2b python mpidist.py

#2 when the backend is NCCL,三种初始化方法TCP，共享文件file,环境变量‘env://’
#TPC.py
import torch.distributed as dist
dist.init_process_group(backend='mpi',init_method='tcp://127.0.0.1:12345',
                        rank=args.rank,world_size=2)
python TCP.py --init_method tcp://127.0.0.1:12345 --rank 0 --world_size 2

#file.py各个进程在共享文件系统中通过该文件进行同步或异步。因此，所有进程必须对该文件具有读写权限。
#说明：若指定为同一文件，则每次训练开始之前，该文件必须手动删除，但是文件所在路径必须存在！
import torch.distributed as dist
dist.init_process_group(backend='nccl',init_method='file:///mnt/nfs/sharedfile',
                        rank=args.rank, world_size=2)
python file.py --init_method file://mnt/nfs/sharedfile --rank 0 --world_size 2
#这里相比于 TCP 的方式麻烦一点的是运行完一次必须更换共享的文件名，或者删除之前的共享文件，不然第二次运行会报错。

#env.py 默认情况下使用的都是环境变量来进行分布式通信，也就是指定 init_method="env://"
#通过在所有机器上设置如下四个环境变量，所有的进程将会适当的连接到 master，获取其他进程的信息，并最终与它们握手(信号)。
#四个环境变量: MASTER_PORT, MASTER_ADDR,WORLD_SIZE,RANK,前两个必须指定,rank0对应的端口和地址
#配合 torch.distribution.launch 使用。
#env.py看上面的
python -m torch.distributed.launch --nproc_per_node=3 --nnode=2 --node_rank=0
--master_addr='127.0.0.1' --master_port=1234 env.py(*args,**kwargs)




#Distributed Modules
net = Net()
device = torch.device('cuda',args.local_rank)
net = net.to(device)
net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[args.local_rank],output_device=args.local_rank)
#1 DistributedDataParallel (分为单节点多GPU，多借点多GPU）
#函数定义
torch.nn.parallel.DistributedDataParallel(module,
                                          device_ids=None,
                                          output_device=None,
                                          dim=0,
                                          broadcast_buffers=True,
                                          process_group=None,
                                          bucket_cap_mb=25,
                                          find_unused_parameters=False,
                                          check_reduction=False)
'''说明：将给定的 module 进行分布式封装， 其将输入在 batch 维度上进行划分，并分配到指定的 devices 上。
module 会被复制到每台机器的每个 GPU 上，每一个模型的副本处理输入的一部分。
在反向传播阶段，每个机器的每个 GPU 上的梯度进行汇总并求平均。与 DataParallel 类似，batch size 应该大于 GPU 总数,并且为GPU的整数倍。
对于模型并行的情况，即一个模型，分散于多个 GPU 上的情况（multi-device module），以及 CPU 模型，device_ids和output_devices
必须为 None，或者为空列表。
要使用该 class，需要先对 torch.distributed 进行初进程组始化，可以通过 torch.distributed.init_process_group() 实现。
该 module 仅在 gloo 和 nccl 后端上可用。
需要与多 workers 的 Dataloader 一同使用
在进程之间，参数永远不会进行 broadcast。该 module 对梯度执行一个 all-reduce 步骤，并假设在所有进程中，可以被 optimizer 以相同的
方式进行更改。 在每一次迭代中，Buffers (BatchNorm stats 等) 是进行 broadcast 的，从 rank 0 的进程中的 module 进行广播，广播到系统的其他副本中。'''

#a) 单节点多GPU
#这种方法不好，默认使用所有GPU
torch.distributed.init_process_group(backend="nccl")
model = DistributedDataParallel(model)

#b)多节点多GPU
torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
model = DistributedDataParallel(model, device_ids=[i], output_device=i])
#为了在每个主机（node）上使用多进程，可以使用 torch.distributed.launch 或 torch.multiprocessing.spawn 来实现。
#注意：torch.nn.parallel.DistributedDataParallel与torch.distribted.init_process_group一起使用, 即开多个terminal
#就可以模拟多机单GPU了
#建议使用每个进程一个GPU的方式Multi-process single_GPU
python torch.cuda.set_device(i)
#程序内
torch.distributed.init_process_group(backend='nccl',
                                     init_method='env://',
                                     rank=0,
                                     world_size=2)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[i],output_device=i)

#2 DistributedDataParallelCPU
torch.nn.parallel.DistributedDataParallelCPU(module)
'''
该 module 支持 mpi 和 gloo 后端。
torch.nn.parallel.DistributedDataParallelCPU与torch.utils.data.distributed.DistributedSampler一起使用
DistributedDataParallelCPU通过在 batch 维度上，对输入进行分割，并分配到特定的设备上，实现模型的并行。
将该 module 复制到每一台机器上，每一个副本处理输入的一部分。在反向传播阶段，每各节点的梯度求平均。
DistributedSampler 将会为每个节点加载一个原始数据集的子集，每个子集的 batchsize 相同。
总的batch-size=node的总数*每个node上的batchsize
'''

#3 DistributedSampler
#dataset:进行采样的总数据，num_replicas：分布式中参与训练的进程数,rank：当前进程序号
torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=None, rank=None)
'''
对数据集进行采样，使之划分为几个子集，不同 GPU 读取的数据应该是不一样的。
一般与 DistributedDataParallel 配合使用。
此时，每个进程可以传递一个 DistributedSampler 
实例作为一个 Dataloader sampler，并加载原始数据集的一个子集作为该进程的输入。
注意：在 DataParallel 中，batch size 设置必须为单卡的 n 倍，但是在 DistributedDataParallel 内，batch size 设置于单卡一样即可。
'''
# 分布式训练示例
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

dataset = your_dataset()
datasampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size_per_gpu, sampler=datasampler)
model = your_model()
model = DistributedDataPrallel(model, device_ids=[local_rank], output_device=local_rank)

#通信方式(torch.distributed支持Collective communication集体通信和Point-to-point通信
#1 Point-to-Point通信
'''
point-to-point通信通过send,recv实现(阻塞等待式）, 或者isend, irecv（异步无阻塞式）
point-to-point使得进程间实现fine-grained通信控制
'''
#send,recv: dst目标rank, src:y源rank，如果没有指定，从任意进程接受数据， group:工作的进程组, tag:匹配当前send和远程recv
torch.distributed.send(tensor, dst, group=<object>, tag=0)
torch.distributed.recv(tensor, src=None, group=<object>, tag=0)

#isend,irecv：异步发送和接受tensor
torch.distributed.isend(tensor, dst, group=<object>, tag=0)
torch.distributed.irecv(tensor, src=None, group=<object>, tag=0)
#is_completed():判断通信是否完成
#wait()对进程上锁，等待通信结束。
"""Blocking point-to-point communication."""
def run(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])
#最后两个进程中的tensor从零都变为1

"""Non-blocking point-to-point communication."""
#wait(),对进程上锁，等待通信结束。在 req.wait() 执行之后，我们可以保证通信已经结束。
def run(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])

#2 collective communication集体通信
'''
1 群体操作，支持同步和异步的方式
2 同步操作（默认）
  当 async_op 设置为 False 时，为同步操作
  当函数返回时，可以保证 collective 操作执行完毕(由于所有的 CUDA 操作都是异步的，因此当为 CUDA 操作时，操作不一定已完成)
3 异步操作
  当 async_op 设置为 True 时，为此模式。
  此时，collective 操作函数返回一个分布式请求对象。通常来说，无需手动创建该对象，且该对象支持两个操作：
    is_completed() ：判断是否执行完毕，若是则返回 True
    wait()：使用这个方法来阻塞这个进程，直到调用的 collective function 执行完毕
  要创建一个组，可以传递一个 rank 的列表给 dist.new_group(group)
  默认情况下，集体操作是执行在所有的进程上的，也被称之为 world（所有的进程）。
'''
#单GPU集体操作
#1 broadcast: 广播该 tensor 到整个 group。若 src 为当前进程，则 tensor 为要发送的数据；若不为当前进程，则 tensor 为要接收的数据。
torch.distributed.broadcast(tensor, src, group=<object>, async_op=False)
#2 scatter:分发 tensor 到组内所有进程，注意与 broadcast 的区别,注意这里group中每个进程收到list中的一个数据，不是全部数据
torch.distributed.scatter(tensor, scatter_list, src, group=<object>, async_op=False)
#3 barrier: 同步所有进程。若 async_op 为 False，或 async 进程是在 wait() 中调用的，则该操作将封锁进程，直到整个组进入该函数。
torch.distributed.barrier(group=<object>, async_op=False)
#4 gather:将一组 tensor 聚集于一个进程。gather_list仅在接收数据的进程中需要设定，为一个尺寸合适的 tensor。
torch.distributed.gather(tensor, gather_list, dst, group=<object>, async_op=False)
#5 all_gather:将 group 中的 tensor 集中到 tensor_list 中。group中每个进程都保存这个list
torch.distributed.all_gather(tensor_list, tensor, group=<object>, async_op=False)
#6reduce:对所有进程内的数据进行归约。但是结果只存储于 dst 进程。tensor:collective 输入输出，该操作是 inplace 的
torch.distributed.reduce(tensor, dst, op=ReduceOp.SUM, group=<object>, async_op=False)
#op
'''
指定归约操作的类型，为 torch.distributed.ReduceOp 枚举类型。支持：
torch.distributed.ReduceOp.SUM
torch.distributed.ReduceOp.PRODUCT
torch.distributed.ReduceOp.MIN
torch.distributed.ReduceOp.MAX
'''
#7 all_reduce:与 reduce 一致，区别在于，所有进程都获取最终结果，inplace 操作。
#tensor:collective 输入输出，该操作是 inplace 的
torch.distributed.torch.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=<object>, async_op=False)

#多GPU集体操作: 一个进程享有多个GPU，使得每个进程运行速度变快, tensor_list中的每个tensor必须位于同一个GPU
#当使用 NCCL 和 Gloo 后端时，如果每个节点上拥有多个 GPU，支持每个节点内的多 GPUs 之间的分布式 collective 操作。
# 要注意到每个进程上的 tensor list 长度都必须相同。
#函数调用时，在传递的列表中的每个 tensor，需要在主机的一个单独的 GPU 上。

#1 all_reduce_multigpu: tensor_list中存放的是当前进程对应的所有GPU中的tensor列表，
# src_tensor可以指定对当前进程中哪个GPU的tensor进行广播,src_tensor 指定的元素（tensor_list[src_tensor]）
#需要保证，对于所有调用该函数的分布式进程中，tensor_list 的长度是一样的。
torch.distributed.broadcast_multigpu(tensor_list, src, group=<object>, async_op=False, src_tensor=0)
#2 all_gather_multigpu
#需要注意，output_tensor_lists 内的每一个元素，尺寸均为 world_size *len(input_tensor_list)。
#input_tensor_list[j] 中索引为 k 的值，对应于 output_tensor_lists[i][k *world_size + j]。
#input_tensor_list:当前进程中，需要进行 broadcast 的 tensors 的列表，每个 tensors 应该位于不同的 GPU 上。
#要注意，所有调用该函数的分布式进程中，input_tensor_list 的长度应该一致。
torch.distributed.all_gather_multigpu(output_tensor_lists, input_tensor_list, group=<object>, async_op=False)
#3 reudce_multigpu:对所有机器上的多个 GPUs 中的 tensors 进行归约。tensor_list 内的每个 tensor 应该位于独立的 GPU 上。
#只有 dst 进程上的 tensor_list[dst_tensor] 对应的 GPU 会接收最后的结果。
torch.distributed.reduce_multigpu(tensor_list, dst, op=ReduceOp.SUM, group=<object>, async_op=False, dst_tensor=0)
#4 all_reduce_multigpu对所有机器上的多个 GPUs 中的 tensors 进行归约。tensor_list 内的每个 tensor 应该位于独立的 GPU 上。
#所有的进程都将获得最终结果。
torch.distributed.all_reduce_multigpu(tensor_list, op=ReduceOp.SUM, group=<object>, async_op=False)
#example: 2 nodes,8GPU 2*8
#node 0:
import torch
import torch.distributed as dist
dist.init_process_group(backend='nccl',
                        init_method='file:///distributed_test',
                        world_size=2,
                        rank=0)
tensor_list = []

for dev_idx in range(torch.cuda.device_count()):
    tensor_list.append(torch.FloatTensor([1])).cuda(dev_idx)
dist.all_reduce_multigpu(tensor_list)
#node1
import torch
import torch.distributed as dist
dist.init_process_group(backend='nccl',
                        init_method='file:///distributed_test',
                        world_size=2,
                        rank=1)
tensor_list = []
for dev_idx in range(torch.cuda.device_count()):
    tensor_list.append(torch.FloatTensor([1])).cuda(dev_idx)
dist.all_reduce_multigpu(tensor_list)
#运行后，所有GPU上的tensor值都为16

#single node multigpu
'''
模型训练阶段:
假设每个主机有 N 个 GPUs，那么需要使用 N 个进程，并保证每个进程单独处理一个 GPU。
因此，需要保证训练代码在单个 GPU 上进行操作，可以用如下代码进行实现：
'''
python torch.cuda.set_device(i)
'''构建模型：其中，i 应该为 0 到 N - 1 之间。'''
torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
train_set = torchvision.datasets.CIFAR10()
train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=122,sampler=train_sampler)
model = DistributedDataParallel(model, device_ids=[i], output_device=i])