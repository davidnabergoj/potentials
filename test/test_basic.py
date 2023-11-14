import pytest
import torch

from potentials.base import PotentialSimple
from potentials.synthetic.gaussian_mixture import GaussianMixture2D
from potentials.synthetic.gaussian.unit import StandardGaussian
from potentials.synthetic.gaussian.diagonal import (
    DiagonalGaussian0,
    DiagonalGaussian1,
    DiagonalGaussian2,
    DiagonalGaussian3,
    DiagonalGaussian4,
    DiagonalGaussian5,
)
from potentials.synthetic.gaussian.full_rank import (
    FullRankGaussian0,
    FullRankGaussian1,
    FullRankGaussian2,
    FullRankGaussian3,
    FullRankGaussian4,
    FullRankGaussian5,
)
from potentials.synthetic.gaussian.block_diagonal import (
    BlockDiagonal0,
    BlockDiagonal1,
    BlockDiagonal2,
    BlockDiagonal3,
    BlockDiagonal4,
    BlockDiagonal5,
)
from potentials.synthetic.funnel import Funnel
from potentials.synthetic.multimodal import (
    DoubleGaussian0,
    DoubleGaussian1,
    TripleGaussian0,
    TripleGaussian1,
    TenRandomlyPositionedGaussians
)

__target_list = [
    StandardGaussian,
    DiagonalGaussian0,
    DiagonalGaussian1,
    DiagonalGaussian2,
    DiagonalGaussian3,
    DiagonalGaussian4,
    DiagonalGaussian5,
    DoubleGaussian0,
    DoubleGaussian1,
    TripleGaussian0,
    TripleGaussian1,
    TenRandomlyPositionedGaussians,
    Funnel
]
__unstable_target_list = [
    BlockDiagonal0,
    BlockDiagonal1,
    BlockDiagonal2,
    BlockDiagonal3,
    BlockDiagonal4,
    BlockDiagonal5,
    FullRankGaussian0,
    FullRankGaussian1,
    FullRankGaussian2,
    FullRankGaussian3,
    FullRankGaussian4,
    FullRankGaussian5,
]  # Won't reliably go above 100 dimensions


@pytest.mark.parametrize('device', [torch.device('cpu'), torch.device('cuda')])
@pytest.mark.parametrize('potential_class', __target_list + __unstable_target_list)
@pytest.mark.parametrize('n_dim', [2, 10, 100, 1000])
@pytest.mark.parametrize('batch_shape', [(2, 3), (5, 4), (7, 19, 2), (11,)])
def test_evaluation(batch_shape, n_dim, potential_class: PotentialSimple, device):
    if device == torch.device("cuda") and not torch.cuda.is_available():
        pytest.skip("Skipping CUDA test")
    if potential_class in __unstable_target_list and n_dim > 100:
        pytest.skip("Not testing inherently numerically unstable target in high dimensions")
    torch.manual_seed(0)
    potential = potential_class(n_dim=n_dim).to(device)
    x = torch.randn(size=(*batch_shape, *potential.event_shape)).to(device)
    u = potential(x)
    assert u.shape == batch_shape
    assert torch.all(~torch.isnan(u))
    assert torch.all(~torch.isinf(u))


@pytest.mark.parametrize('device', [torch.device('cpu'), torch.device('cuda')])
@pytest.mark.parametrize('potential_class', __target_list + __unstable_target_list)
@pytest.mark.parametrize('n_dim', [2, 10, 100, 1000])
@pytest.mark.parametrize('batch_shape', [(2, 3), (5, 4), (7, 19, 2), (11,)])
def test_sampling(batch_shape, n_dim, potential_class: PotentialSimple, device):
    if device == torch.device("cuda") and not torch.cuda.is_available():
        pytest.skip("Skipping CUDA test")
    if potential_class in __unstable_target_list and n_dim > 100:
        pytest.skip("Not testing inherently numerically unstable target in high dimensions")
    torch.manual_seed(0)
    potential = potential_class(n_dim=n_dim).to(device)
    x = potential.sample(batch_shape)
    assert x.shape == (*batch_shape, *potential.event_shape)
    assert torch.all(~torch.isnan(x))
    assert torch.all(~torch.isinf(x))


@pytest.mark.parametrize('device', [torch.device('cpu'), torch.device('cuda')])
@pytest.mark.parametrize('potential_class', __target_list + __unstable_target_list)
@pytest.mark.parametrize('batch_shape', [(2, 3), (5, 4), (7, 19, 2), (11,)])
def test_fixed_n_dim_evaluation(batch_shape, potential_class, device):
    if device == torch.device("cuda") and not torch.cuda.is_available():
        pytest.skip("Skipping CUDA test")
    torch.manual_seed(0)
    potential = potential_class().to(device)
    x = torch.randn(size=(*batch_shape, *potential.event_shape)).to(device)
    u = potential(x)
    assert u.shape == batch_shape
    assert torch.all(~torch.isnan(u))
    assert torch.all(~torch.isinf(u))


@pytest.mark.parametrize('device', [torch.device('cpu'), torch.device('cuda')])
@pytest.mark.parametrize('potential_class', __target_list + __unstable_target_list)
@pytest.mark.parametrize('batch_shape', [(2, 3), (5, 4), (7, 19, 2), (11,)])
def test_fixed_n_dim_sampling(batch_shape, potential_class, device):
    if device == torch.device("cuda") and not torch.cuda.is_available():
        pytest.skip("Skipping CUDA test")
    torch.manual_seed(0)
    potential = potential_class().to(device)
    x = potential.sample(batch_shape)
    assert x.shape == (*batch_shape, *potential.event_shape)
    assert torch.all(~torch.isnan(x))
    assert torch.all(~torch.isinf(x))
