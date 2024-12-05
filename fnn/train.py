import torch
from torch.cuda import is_available
from torchvision import transforms
import torchvision
from torch import nn
import sys
import os
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from network import mynn,num_nn
from read_data import MyDataset

# with tensorBoard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs')

# define dataset & DataLoader
mytrans = transforms.ToTensor()
#train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True,download=True,transform=mytrans)
train_set = MyDataset("./0","./1")
myloader = DataLoader(dataset=train_set, batch_size=256, shuffle=True, drop_last=False)

# define network  
network = num_nn()
#network = mynn()

# define loss
loss_fn = nn.CrossEntropyLoss()

# define optimizer
optimizer = torch.optim.Adam(network.parameters())

# Use cuda
if torch.cuda.is_available():
    print("Cuda available!")
    network = network.cuda()
    loss_fn = loss_fn.cuda()

# epoch
epoch = 1000

for i in range(epoch):
    loss_temp = 0.0
    for data in myloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = network(imgs)
        loss = loss_fn(outputs,targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_temp = loss.item()
    writer.add_scalar('loss_func',loss_temp,i)
    writer.flush()
    print(f"epoch is {i}, loss is {loss_temp}")

writer.close()



