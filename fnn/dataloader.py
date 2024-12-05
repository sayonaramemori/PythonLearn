import torchvision


from torchvision import transforms
from torch.utils.data import DataLoader

mytrans = transforms.ToTensor()

train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True,download=True,transform=mytrans)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False,download=True,transform=mytrans)

myloader = DataLoader(dataset=train_set, batch_size=64, shuffle=True, drop_last=False)

