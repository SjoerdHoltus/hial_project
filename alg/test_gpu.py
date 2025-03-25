# Add these lines at the very top of your file, before any other imports
import os
# Fix for OpenMP runtime error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
# torch.backends.cudnn.benchmark = True     # Enable cuDNN auto-tuner

# Check GPU availability with more detailed information
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU")