import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# Define a dataset
# Specify its root_dir
# With transform  
mytrans = transforms.ToTensor()
train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True,download=True,transform=mytrans)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False,download=True,transform=mytrans)



logs = SummaryWriter("logs")

for i in range(20):
    img, label = train_set[i]
    logs.add_image("pic",img,label)

logs.close()
