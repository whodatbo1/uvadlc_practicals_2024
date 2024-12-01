import torch
import torch.nn.functional as F
torch.manual_seed(42)

probs = torch.rand(10)

probs_scaled = probs * 100

probs = F.softmax(probs, dim=-1)
probs_2 = F.softmax(probs_scaled, dim=-1)

print(f'probs: {probs}')
print(f'probs_2: {probs_2}')
