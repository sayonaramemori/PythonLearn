### Anaconda cmd
```shell 
conda env list
conda list  # In env
```

### Install Pytorch  


### Tensor  
```python 
a = torch.randn(2,3) # A tensor with 2 row and 3 colume
isinstance(a,torch.FloatTensor) # True

# Scalar dimension 0
a = torch.tensor(1.0) #f32 or int
a.shape # torch.Size([])
a.size(n) # n is the index of dimension
a.numel() # numbers in tensor
a.dim()   # dimension of a 

# Vector dimension 1
a = torch.tensor([1.1])  # default f32
b = torch.FloatTensor(1) # Parameter is size of a vector
```

#### From numpy  
```python 
a = np.array([2,3.3])
b = torch.from_numpy(a)
```

#### From List  
```python
torch.tensor([2,3.4])
tensor([2.00,5.00])
torch.FloatTensor([3.,7.])   # It's better to receive shape Parameter instead of raw datas
torch.tensor([[2,3.4],[3,.5]])
```

#### Uninitialized  
```python  
torch.Tensor(1,2,3) #default float32
torch.set_default_dtype(torch.float64) # To change the default settings
torch.IntTensor(d1,d2,d3)
torch.FloatTensor(d1,d2,d3)
torch.DoubleTensor(d1,d2,d3)
```

#### From Methods  
```python 
torch.full([dims,],val);
torch.full([],7) # dim 0
torch.full([8],7) # dim 1

torch.arange(0,10,step=1) #[start,end)

torch.linspace(0,10,steps=4)
torch.logspace(0,-1,steps=10)

torch.ones(3,3)
torch.zeros(3,3)
torch.eye(3,3) # dialog

torch.ones_like(a) # same shape
```


### Index & Slice on Tensor
```python
a = torch.rand(4,3,28,28)
a[0].shape # 3,28,28
a[0,0].shape # 28,28
a[0,0,0,0] # A scalar

a[:2].shape #2,3,28,28
a[:2,:1,:,:].shape #2,1,28,28
a[:2,-1:,:,:].shape #2,1,28,28
a[:,1,...].shape # 4,28,28

# Subsampling 
a[:,:,::2,::2].shape #4,3,14,14

a.index_select(dim_index,[index,])
```

### Dimension Transform  
```python 
a = torch.rand(4,1,28,28)
# Or reshape
a.view(4,28*28) # Promise the numel is constant

# Reinterpret the meanings of data
a.unsqueeze(0).shape # 1,4,1,28,28
a.unsqueeze(-1).shape # 4,1,28,28,1
a.squeeze().shape # All dim val == 1 are removed

# Expanding 
a = torch.rand(4,32,14,14)
b = torch.rand(1,32,1,1)
b.expand(4,32,14,14)

# Transpose, exchange dimension
a.transpose(d1,d2)
a2 = a.transpose(1,3).contiguous().view(4,32*14*14).view(4,32,14,14).transpose(1,3)
torch.all(torch.eq(a,a2)) #true

# Manipulate Dimension
a.permute(0,2,3,1).shape # 4,14,14,32
```

### Broadcasting  
> Auto-Expanding without copying data

1. Insert 1 dim ahead  
2. Expand dims with size 1 to same size  


#### Why Broadcasting?  
1. For example [4,32,8] + [5.0]

#### Is it Broadcasting-Able?
> Match from last dim

1. If current dim==1, expand to same
2. If has no dim, insert one dim and expand to same
3. Otherwise, it's not Broadcasting-Able

### Split and Merge for Tensor
```python
a = torch.rand(4,32,8)
b = torch.rand(5,32,8)

# Only the target dim can be different
torch.cat([a,b],dim=0).shape # 9,32,8

c = torch.rand(4,32,8)
# create a new dim, so dim of a&c should be the same
e = torch.stack([a,c],dim=0).shape #[2,4,32,8]
f,g = torch.split([1,1],dim=0)
#f,g = torch.split(1,dim=0)
f.shape # [4,32,8]
```

### Numeric Operation on Tensor
```python 
a = torch.rand(3,4)
b = torch.rand(4)
a+b
torch.add(a,b)
a*b 
torch.mul(a,b)
a/b
torch.div(a,b)

# Mat mul
a = torch.ones(2,2)
b = a
torch.matmul(a,b)
a@b
e = torch.rand(4,3,28,64)
f = torch.rand(4,3,64,55)
e@f #shape is 4,3,28,55

# Power Exp Log element-wise
aa=a**2
a.pow(2)
aa.sqrt()
aa**(0.5)
torch.exp(aa)
torch.log(aa)

# clamp
aa.clamp(0,10)
```

### Statistics Operation on Tensor  
```python 
a = torch.ones(8)
a.min()
a.max()
a.mean()
a.prod() # production
a.sum()

a.norm(1) # 1范数 sum(abs(ele))
a.norm(2) # sqrt(sum(ele**2))

a.argmax() # return the index(flatten) of the max value
a.argmin() # return the index(flatten) of the min value

a.argmax(dim=n) # return a vector containing the index of that dimension
```

### Dataset  
> A class manages all needed datas  
> Support operator+ for merging two dataset  
```python
from torch.utils.data import Dataset

class MyData(Dataset):
    def __init__(self,data_dir,label_dir,args):
        # Read datas from directory and organized them well for
        # indexing by __getitem__
        self.data = data_dir
        self.label = label_dir
        ...
    def __getitem__(self,index):
        ...
        return data,label
    def __len__(self):
        return data.length
```

### Tensorboard  
> `conda install tensorboard` to install  
> Run `tensorboard --logdir=logs --port=6000` to visualize the logs  

```python 
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("logs")
# writer.add_image()
for i in range(199):
    writer.add_scalar(tag='loss',scalar_value=i,global_step=i)
writer.close
```

### Transforms in TorchVision  
> A Operator to Convert data types to fulfill the training requirements.  
```python 
from torchvision import transforms
# Resize Compose RandomCrop
tensor_tran_operator = transforms.ToTensor()
composed_operator = transforms.Compose([tensor_tran_operator,transforms.Resize])
tensor_img = tensor_tran_operator(image_type)
```

### DataLoader  
```python 
import torchvision
from torch.utils.data import DataLoader
test_data = torchvision.datasets.CIFAR10("./dataset",train=False,transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=10,shuffle=True,num_workers=0)
```

### torch.nn  
```python 
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

class myNN(nn):
    def __init__(self):
        super(myNN,self).__init__()
        self.model = Sequential(
            Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0),
            MaxPool2d(2),
            Conv2d(6,64,5,padding=1),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )

    def forward(self,x):
        return self.model(x)
```

### Loss & Optim  
```python 
mynn = myNN()
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(mynn.parameters(),lr=1e-3)
for batch in dataloader:
    imgs, targets = batch
    outputs = myNN(imgs)                # predict
    result_loss = loss(outputs,targets) # calculate loss
    optim.zero_grad()                   # Set zero for all grad
    result_loss.backward()              # calculate grad here
    optim.step()                        # update parameters in NN
```

### Model Save  
```python  
# dict, only parameter
torch.save(yourNN.state_dict(),"my_nn.pth")

# load dict
mynn_dict = torch.load("my_nn.pth")
mynn = myNN()
mynn.load_state_dict(mynn_dict)
```

### Use GPU
> Only `NN`, `LossFn` and `Data with label` need to be transported to CUDA  
```python 
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
ant_bee_classification_nn = AntBeeNN().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)

for i in range(epoch):
    print("Epoch ",i)
    for data in train_dataloader:
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = ant_bee_classification_nn(imgs)
        loss = loss_fn(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
