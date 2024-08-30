from pathlib import Path
from typing import Union, Tuple

import torch

from potentials.base import Potential


class Rosenbrock(Potential):
    def __init__(self, event_shape: Union[int, Tuple[int, ...]] = (10,), scale: float = 10.0):
        super().__init__(event_shape=event_shape)
        self.scale = scale

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = torch.flatten(x, start_dim=-len(self.event_shape), end_dim=-1)
        first_term_mask = torch.arange(self.n_dim) % 2 == 0
        second_term_mask = ~first_term_mask
        return torch.sum(
            torch.add(
                self.scale * (x_flat[..., first_term_mask] ** 2 - x_flat[..., second_term_mask]) ** 2,
                (x_flat[..., first_term_mask] - 1) ** 2
            ),
            dim=1
        )

    @property
    def mean(self):
        if tuple(self.event_shape) == (100,) and self.scale == 10.0:
            path = Path(__file__).parent.parent / 'true_moments' / f'rosenbrock_moments.pt'
            if path.exists():
                return torch.load(path)[0]
            return torch.tensor([
                0.9984, 1.4949, 1.0021, 1.5041, 1.0027, 1.5069, 0.9996, 1.4978, 1.0032,
                1.5112, 0.9950, 1.4879, 0.9995, 1.5014, 1.0018, 1.5072, 0.9987, 1.4968,
                1.0006, 1.4967, 1.0016, 1.5068, 0.9980, 1.4960, 0.9990, 1.4978, 0.9953,
                1.4867, 0.9995, 1.4979, 0.9950, 1.4842, 1.0027, 1.5045, 1.0032, 1.5070,
                1.0017, 1.5049, 0.9977, 1.4948, 1.0025, 1.5081, 1.0008, 1.5020, 1.0032,
                1.5037, 0.9972, 1.4917, 0.9966, 1.4946, 1.0021, 1.5043, 0.9979, 1.4949,
                0.9965, 1.4906, 1.0017, 1.5055, 0.9966, 1.4921, 1.0043, 1.5143, 1.0027,
                1.5068, 0.9986, 1.4943, 1.0012, 1.5015, 1.0028, 1.5071, 1.0026, 1.5036,
                1.0022, 1.5047, 1.0000, 1.4993, 0.9967, 1.4934, 1.0000, 1.5025, 1.0026,
                1.5042, 0.9996, 1.4969, 1.0030, 1.5037, 1.0023, 1.5039, 1.0041, 1.5117,
                0.9973, 1.4969, 0.9963, 1.4910, 0.9954, 1.4892, 1.0032, 1.5075, 0.9974,
                1.4981
            ])
        return super().mean

    @property
    def second_moment(self):
        if tuple(self.event_shape) == (100,) and self.scale == 10.0:
            path = Path(__file__).parent.parent / 'true_moments' / f'rosenbrock_moments.pt'
            if path.exists():
                return torch.load(path)[1]
            return torch.tensor([
                1.4946, 4.7206, 1.5038, 4.7998, 1.5067, 4.8607, 1.4983, 4.7665, 1.5112,
                4.8907, 1.4885, 4.7134, 1.5016, 4.8135, 1.5072, 4.8636, 1.4962, 4.7733,
                1.4973, 4.7428, 1.5065, 4.8555, 1.4961, 4.7844, 1.4974, 4.7762, 1.4864,
                4.7298, 1.4979, 4.7911, 1.4844, 4.6686, 1.5051, 4.8163, 1.5069, 4.8223,
                1.5048, 4.8214, 1.4951, 4.7586, 1.5076, 4.8529, 1.5014, 4.8149, 1.5043,
                4.8092, 1.4918, 4.7380, 1.4947, 4.7775, 1.5047, 4.8450, 1.4950, 4.7872,
                1.4901, 4.7427, 1.5051, 4.8384, 1.4927, 4.7605, 1.5149, 5.0037, 1.5066,
                4.8262, 1.4945, 4.7642, 1.5018, 4.8078, 1.5073, 4.8402, 1.5030, 4.7844,
                1.5042, 4.8380, 1.4995, 4.8093, 1.4931, 4.7657, 1.5026, 4.8292, 1.5045,
                4.8126, 1.4973, 4.7791, 1.5044, 4.7838, 1.5040, 4.8092, 1.5113, 4.8765,
                1.4977, 4.7916, 1.4901, 4.7216, 1.4895, 4.7160, 1.5083, 4.8565, 1.4983,
                4.7773
            ])
        return super().second_moment


if __name__ == '__main__':
    u = Rosenbrock((100,))
    print(u.mean.shape)
    print(u.second_moment.shape)
