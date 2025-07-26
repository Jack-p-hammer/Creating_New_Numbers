import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())

print(torch.version.cuda)     # Should print your CUDA version (e.g., '11.8')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")