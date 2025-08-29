import torch
import numpy as np
import random

# For reproducibility
torch.manual_seed(42)       # CPU
torch.cuda.manual_seed(42)  # Current GPU
torch.cuda.manual_seed_all(42)  # All GPUs (if multi-GPU)
np.random.seed(42)
random.seed(42)

# For deterministic behavior (slower, but reproducible)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
