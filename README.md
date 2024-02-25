# Bayex: Minimal Bayesian Optimization in JAX
[![tests](https://github.com/alonfnt/bayex/actions/workflows/tests.yml/badge.svg)](https://github.com/alonfnt/bayex/actions/workflows/tests.yml)

<p align="center">
    <img src="https://github.com/alonfnt/bayex/assets/38870744/ffb920ed-f347-4185-9abe-24ec2d0a22f1" height="300">
    <img src="https://github.com/alonfnt/bayex/assets/38870744/882fecc7-bc30-4267-ad1d-687fdbbe2cdc" height="300">
</p>

[**Installation**](#installation)
| [**Usage**](#usage)
| [**Contributing**](#contributing)
| [**License**](#license)

Bayex is a lightweight Bayesian optimization library designed for efficiency and flexibility, leveraging the power of JAX for high-performance numerical computations.
This library aims to provide an easy-to-use interface for optimizing expensive-to-evaluate functions through Gaussian Process (GP) models and various acquisition functions. Whether you're maximizing or minimizing your objective function, Bayex offers a simple yet powerful set of tools to guide your search for optimal parameters.

## Installation<a id="installation"></a>
Bayex can be installed using [PyPI](https://pypi.org/project/bayex/) via `pip`:
```
pip install bayex
```
or from GitHub directly
```
pip install git+git://github.com/alonfnt/bayex.git
```

Likewise, you can clone this repository and install it locally

```bash
git clone https://github.com/alonfnt/bayex.git
cd bayex
pip install -r requirements.txt
```

## Usage<a id="usage"></a>
Using Bayex is quite simple despite its low level approach:
```python
import jax
import numpy as np
import bayex

def f(x):
    return -(1.4 - 3 * x) * np.sin(18 * x)

domain = {'x': bayex.domain.Real(0.0, 2.0)}
optimizer = bayex.Optimizer(domain=domain, maximize=True, acq='PI')

# Define some prior evaluations to initialise the GP.
params = {'x': [0.0, 0.5, 1.0]}
ys = [f(x) for x in params['x']
opt_state = optimizer.init(ys, params)

# Sample new points using Jax PRNG approach.
ori_key = jax.random.key(42)
for step in range(20):
    key = jax.random.fold_in(ori_key, step)
    new_params = optimizer.sample(key, opt_state)
    y_new = f(**new_params)
    opt_state = optimizer.fit(opt_state, y_new, new_params)
```

with the results being saved at `opt_state`.

## Contributing<a id="contributing"></a>
We welcome contributions to Bayex! Whether it's adding new features, improving documentation, or reporting issues, please feel free to make a pull request or open an issue.

## License<a id="license"></a>
Bayex is licensed under the MIT License. See the ![LICENSE](LICENSE) file for more details.
