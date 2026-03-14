import torch
import numpy as np

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.__version__)
print(np.__version__)