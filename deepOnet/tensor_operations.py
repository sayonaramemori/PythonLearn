import torch
import torch.nn as nn
import numpy as np
import pin_random

a = torch.tensor([[1,2,3],[4,5,6]])
b = torch.tensor([[1,2,3],[4,5,6]])
print(a*b)

data = np.arange(100)
test_size = 0.2

# shuffle indices
indices = np.arange(len(data))
np.random.shuffle(indices)

split = int(len(data) * (1 - test_size))
train_idx, test_idx = indices[:split], indices[split:]

train_data, test_data = data[train_idx], data[test_idx]

print("Train:", train_data)
print("Test:", test_data)

