from torch import nn

class TestNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10,20)
        # self.layer2 = nn.
    def forward(self,x):
        return self.layer1(x)

if __name__ == "__main__":
    test = TestNN()
    print(test._modules)
    print(test._modules['layer1'].weight)
    print(test._modules['layer1'].bias)

