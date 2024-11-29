from gpt import RMSNorm
import torch

rms = RMSNorm(dim=10).forward(torch.randn(10))
print(rms)