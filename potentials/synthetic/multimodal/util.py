import numpy as np
import torch


def generate_mode_locations(n_components, n_dim, max_scale: int = 10.0, seed: int = 0):
    rng = np.random.RandomState(seed)
    return torch.as_tensor(rng.randn(n_components, n_dim).astype(np.float32) * n_dim * max_scale, dtype=torch.float)
