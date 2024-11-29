import torch
from rotary_embedding_torch import RotaryEmbedding

# instantiate the positional embedding in your transformer and pass to all your attention layers

rotary_emb = RotaryEmbedding(dim = 32)

# mock queries and keys - dimensions should end with (seq_len, feature dimension), and any number of preceding dimensions (batch, heads, etc)

q = torch.randn(1, 1, 16, 32) # queries - (batch, heads, seq len, dimension of head)
k = torch.randn(1, 1, 16, 32) # keys

# apply the rotations to your queries and keys after the heads have been split out, but prior to the dot product and subsequent softmax (attention)

q = rotary_emb.rotate_queries_or_keys(q)
k = rotary_emb.rotate_queries_or_keys(k)

print(q)
print(k)

# then do your attention with your queries (q) and keys (k) as usual