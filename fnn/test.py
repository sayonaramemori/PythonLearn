import torch

a = torch.tensor([1.2,9.9,8])
b = torch.tensor([1.2,9.9,8])

print(a.shape)
print(a.size())
print(a.type())
print(torch.stack([a,b],dim=0).shape)
