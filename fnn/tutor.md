### Install TensorBoard  
```
conda install -y -c conda-forge tensorboard

```

### Get started   
```
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs')

for i in range(100):
    writer.add_scalar('y=x',i,i)

writer.close()
```


### Basic Usage  
```
tensorboard --logdir=runs --bind_all
```
