import torch

torch.cuda.device_count()
i = 0
torch.device(f'cuda:{i}')
def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
def try_all_gpus():
    devices = [torch.device(f'cuda:{i}')  for i in range(torch.cuda.device_count())]
    return devices if devices else  torch.device('cpy')
x = torch.tensor([1,2,3], device=torch.device('cuda:0'))
y = torch.tensor([1,2,3],device=try_gpu())
print(x.device)
a = torch.tensor([1,2,3], device=torch.device('cuda:0'))
b = torch.tensor([1,2,3],device=torch.device('cpu'))
c = a + b.cuda(0)
d = a + b.clone().to(a.device)
print(c,d)

def allreduce(data):
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    for i in range(1, len(data)):
        data[i] = data[0].to(data[i].device)

self.sensor_transform = [(carla.Transform(carla.Location(x=0, z=2.5)), Attachment.Rigid),
                                 (carla.Transform(carla.Location(x=0, z=2.5)), Attachment.Rigid),
                                 (carla.Transform(carla.Location(x=2.0,z=1.0),carla.Rotation(pitch=5)),Attachment.Rigid),]




