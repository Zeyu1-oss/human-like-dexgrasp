import torch
print("Before any allocation:")
print(torch.cuda.memory_summary())

x = torch.randn(1000, 1000, device='cuda')
print("\nAfter allocation:")
print(torch.cuda.memory_summary())

del x
torch.cuda.empty_cache()
print("\nAfter cleanup:")
print(torch.cuda.memory_summary())