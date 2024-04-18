from potentials.synthetic.funnel import FunnelBase
from potentials.synthetic.multimodal import SimpleTripleGaussian1D


class ChoppedFunnel(FunnelBase):
    """
    x0 ~ GMM(mu1, mu2, mu3, sigma1, sigma2, sigma3)
    xi ~ N(0, exp(x0) / 2); std(xi) = exp(x0 / 2)
    """

    def __init__(self, n_dim: int):
        super().__init__(n_dim, SimpleTripleGaussian1D())
