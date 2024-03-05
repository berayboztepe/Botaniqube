import torch
import numpy as np
import random

seed_value = 42
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value) # if use multi-GPU
np.random.seed(seed_value)
random.seed(seed_value)