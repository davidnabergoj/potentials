# Benchmark potentials in PyTorch

This package provides potential functions to benchmark sampling and optimization algorithms.

Synthetic potentials include:
* Standard Gaussian
* Diagonal Gaussian (several variants)
* Block diagonal Gaussian (several variants)
* Full rank Gaussian (several variants)
* Multimodal (several variants)
* Funnel
* Rosenbrock

Real-world potentials include various model posteriors:
* Eight schools
* German credit
* Sparse German credit
* Radon (varying intercepts)
* Radon (varying slopes)
* Radon (varying intercepts and slopes)
* Synthetic item response theory
* Stochastic volatility model

Image-based potentials include $\phi^4$ lattice field theory targets with varying lattice sizes.

## Usage instructions
The main components in the package are callable `Potential` objects.
They can be used to compute the negative unnormalized log probability density.
In most synthetic examples, `Potential` objects also provide a `sample` function which lets us draw true samples from their underlying distributions.
We provide an example for the funnel potential: 

```python
import torch
from potentials.synthetic.funnel import Funnel

torch.manual_seed(0)

# Create the potential object
potential = Funnel()

batch_shape = (50,)

# Randomly generate some input data
x = torch.randn(size=(*batch_shape, *potential.event_shape)) / 10

# Compute the potential value at input points
value = potential(x)  # value.shape == (50,)

# Draw 50 samples from the underlying distribution
x_new = potential.sample(batch_shape)  # x_new.shape == (50, 100)
```

## Installing

This package requires Python version 3.7 or greater.

Install the package directly from Github:
```
pip install git+https://github.com/davidnabergoj/potentials.git
```

To alternatively configure the package for local development, clone the repository and install dependencies as follows:

```
git clone git@github.com:davidnabergoj/potentials.git
pip install -r requirements.txt
```

## Citation
If you use this code in your work, we kindly ask that you cite the accompanying paper:

```

```