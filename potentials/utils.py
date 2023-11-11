import torch


def get_batch_shape(x: torch.Tensor, event_shape: torch.Size):
    return x.shape[:-len(event_shape)]


def unsqueeze_to_batch(x: torch.Tensor, batch_shape: torch.Size):
    return x[(None,) * len(batch_shape)]


def sum_except_batch(x: torch.Tensor, batch_shape: torch.Size):
    sum_dims = tuple(range(len(batch_shape), len(x.shape)))
    return torch.sum(x, dim=sum_dims)


def sample_from_gamma(sample_shape, gamma_shape: float = 0.5, gamma_scale: float = 1.0, seed: int = 0):
    torch.random.fork_rng()
    torch.manual_seed(seed)
    r = 1 / gamma_scale
    return torch.distributions.Gamma(concentration=gamma_shape, rate=r).sample(sample_shape)
