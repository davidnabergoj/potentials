import torch

from potentials.base import Potential
import numpy as np
from scipy.stats import t as student_t

from potentials.utils import get_batch_shape, sum_except_batch


class StudentT(Potential):
    def __init__(self, df: float, event_shape=(100,), random_state: int = 0):
        super().__init__(event_shape)
        assert df > 1.0
        self.df = df
        self.rv = student_t(self.df)
        self.rv.random_state = np.random.RandomState(seed=random_state)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = get_batch_shape(x, self.event_shape)
        return sum_except_batch(torch.tensor(self.rv.logpdf(x.numpy())).float(), batch_shape)

    def sample(self, sample_shape: tuple) -> torch.Tensor:
        sample_size = int(torch.prod(torch.as_tensor(sample_shape)) * self.n_dim)
        return torch.tensor(self.rv.rvs(size=sample_size)).float().view(*sample_shape, *self.event_shape)

    @property
    def mean(self) -> torch.Tensor:
        return torch.zeros(*self.event_shape)

    @property
    def variance(self) -> torch.Tensor:
        if 1 < self.df <= 2:
            return torch.full(size=self.event_shape, fill_value=torch.inf)
        elif self.df > 2:
            return torch.full(size=self.event_shape, fill_value=self.df / (self.df - 2))
        else:
            raise ValueError


class StudentT0(StudentT):
    def __init__(self, event_shape=(100,)):
        super().__init__(1.5, event_shape)


class StudentT1(StudentT):
    def __init__(self, event_shape=(100,)):
        super().__init__(2.5, event_shape)
