import torch

if torch.cuda.is_available():
    print(torch.__version__)
    print(torch.cuda.get_device_name())
else:
    print("cuda is not available")

torch.set_default_dtype(torch.float64)
a = torch.DoubleTensor(1,23)
b = torch.FloatTensor(1,23)
c = torch.Tensor(1,23)
d = torch.arange(0,10.0)
e = torch.FloatTensor(4,3,9,9)

data = [[
    [1,2,3],
    [4,5,6]
]]

print(a.dtype)
print(b.dtype)
print(c.dtype)
print(d.dtype)
print(e.view(2,2,3,9*9).shape)
print(e.unsqueeze(0).shape)
print(torch.ones_like(e).shape)

f = torch.Tensor(data)
print(f.expand(2,2,2,3))
print(f.expand(8,9,2,3).shape)


