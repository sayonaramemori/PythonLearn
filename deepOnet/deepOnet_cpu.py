import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
from read_xls import get_data_csv
from read_toml import *
import pin_random

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, layers, act=nn.GELU):
        super().__init__()
        net = [nn.Linear(d_in, 32),act(),nn.Linear(32, d_hidden), act()]
        for _ in range(layers-2):
            net += [nn.Linear(d_hidden, d_hidden), act()]
        net += [nn.Linear(d_hidden, 32),act()]
        net += [nn.Linear(32, d_out)]
        self.net = nn.Sequential(*net)
    def forward(self, x): 
        return self.net(x)

class FlotationDataset():
    def __init__(self,train=True):
        FlotationDataset.data = [i for i in get_data_csv()]
        test_size = 0.2
        indices = np.arange(len(FlotationDataset.data))
        np.random.shuffle(indices)
        split = int(len(FlotationDataset.data) * (1 - test_size))
        train_idx, test_idx = indices[:split], indices[split:]
        FlotationDataset.train_data = [FlotationDataset.data[i] for i in train_idx]
        FlotationDataset.test_data = [FlotationDataset.data[i] for i in test_idx]
        self.train = train
        print(f"size of train is {len(FlotationDataset.train_data)}")
    def __getitem__(self,index):
        if self.train:
            return FlotationDataset.train_data[index]
        else:
            return FlotationDataset.test_data[index]
    def __len__(self):
        if self.train:
            return len(FlotationDataset.train_data)
        else:
            return len(FlotationDataset.test_data)


class DeepONet(nn.Module):
    def __init__(self, d_branch_in, d_trunk_in, width=64, modes=16, outputs=1):
        super().__init__()
        self.branch = MLP(d_branch_in, width, modes, layers=5)   # encodes u(·)
        self.trunk  = MLP(d_trunk_in,  width, modes, layers=4)   # encodes y=(x,t)
        self.head   = nn.Linear(modes, outputs)                  # e.g., ux,uy,uz,p,alpha_g,k,eps
    def forward(self, branch_in, trunk_in):
        B = self.branch(branch_in)            # [batch_u, modes]
        T = self.trunk(trunk_in)              # [batch_y, modes]
        # pair every u with every y (operator eval)
        # assume we pass matching pairs: [batch, modes] ⊙ [batch, modes]
        z = B * T
        return self.head(z)

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    # device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    # print(test._modules['branch'].weight)
    # print(test._modules['trunk'])
    fd = FlotationDataset(train=True)
    fd_test = FlotationDataset(train=False)
    train_dataloader = DataLoader(fd,batch_size=batchSize,shuffle=True)
    test_dataloader = DataLoader(fd_test,batch_size=512,shuffle=True)
    test_data_size = len(fd_test)
    for i in range(5):
        donn = DeepONet(3,1).to(device) # print(donn._modules)
        loss_fn = nn.MSELoss().to(device)
        optimizer = optim.Adam(donn.parameters(),lr=lr)
        decayGamma = 0.99+i*0.0016
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decayStepSize, gamma=decayGamma)
        writer = SummaryWriter(log_dir)

        step = 0
        for i in range(20000):
            total_loss = 0.0
            count = 0
            for data in train_dataloader:
                bi,ti,labels = data
                bi = bi.to(device)
                ti = ti.to(device)
                labels = labels.to(device)
                outputs = donn(bi,ti)
                loss = loss_fn(outputs,labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if log_on:
                    writer.add_scalar(f'{log_train_legend} gamma{decayGamma} batchSize{batchSize}',loss.item(),step)
                total_loss += loss.item()
                count += 1
                step += 1
            scheduler.step()
            if i%100==0 :
                print(f"Loss is {total_loss/count}, Step at {i}")
                with torch.no_grad():
                    for data in test_dataloader:
                        bi,ti,labels = data
                        bi = bi.to(device)
                        ti = ti.to(device)
                        labels = labels.to(device)
                        outputs = donn(bi,ti)
                        test_loss = loss_fn(outputs,labels)
                        print("TEST LOSS is ",test_loss.item())
                        if log_on:
                            # writer.add_scalar(log_test_legend,test_loss.item(),step)
                            writer.add_scalar(f'{log_test_legend} gamma{decayGamma} batchSize{batchSize}',test_loss.item(),step)



