import torch
from torch import nn


class mynn(nn.Module):
    def __init__(self):
        super().__init__()
        self.model= nn.Sequential(
            nn.Conv2d(3,32,3,1,1),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,1,1),
            nn.MaxPool2d(2),
            nn.Conv2d(64,32,3,1,1),
            nn.Flatten(),
            nn.Linear(32*64,64),
            nn.Linear(64,10))
    def forward(self, x):
        res = self.model(x)
        return res

class num_nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(3,32,3,1,1),
            nn.Conv2d(32,64,3,1,1),
            nn.Conv2d(64,32,3,1,1),
            nn.Flatten(),
            nn.Linear(32*125*125,64),
            nn.Linear(64,2)
        )
    def forward(self,x):
        res = self.model(x)
        return res

if __name__ == "__main__":
    input = torch.ones((64,3,32,32))
    output = mynn()
    print(output.forward(input).shape)
    input = torch.ones(64,3,250,250)
    output = num_nn()
    print(output.forward(input).shape)

