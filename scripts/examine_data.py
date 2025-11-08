import torch

# Load the .pth file
data = torch.load(
    "/mnt/g/Projects/RICS/PostProcess/data/features/09c1414f1b_feat.pth", map_location="cpu")

# Check type
print(type(data))

# If it's a dict, check keys
if isinstance(data, dict):
    print(data.keys())

# Inspect the content
# For example, if it's a tensor
if isinstance(data, torch.Tensor):
    print(data.shape)
    print(data.dtype)
    print(data[:5])  # first few entries
