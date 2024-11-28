import numpy as np
import matplotlib.pyplot as plt

def plot_embeddings(embeddings, title):
    plt.figure(figsize=(10, 5))
    plt.imshow(embeddings, aspect='auto', origin='lower')
    plt.colorbar()
    plt.title(title)
    plt.show()

def get_absolute_position_embeddings(T, head_dim):
    freqs = 1.0 / (10000 ** (np.arange(0, head_dim, 2) / head_dim))
    # Add extra dimension to freqs
    freqs = freqs[:, np.newaxis]
    return freqs

absolute_position_embeddings = get_absolute_position_embeddings(100, 100)
plot_embeddings(absolute_position_embeddings, "Absolute Position Embeddings")