<p align="center">
    <img src="https://github.com/user-attachments/assets/a761373e-2b34-46a5-9176-201f9d5c5c54" style="width: 640px; height: auto;">
</p>

[![tests](https://github.com/alonfnt/bayex/actions/workflows/tests.yml/badge.svg)](https://github.com/alonfnt/bayex/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/bayex/badge/?version=latest)](https://bayex.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/bayex.svg)](https://pypi.org/project/bayex/)

>[!NOTE]
>Bayex is currently a minimal, personally developed implementation that requires further development for broader application. If you're interested in engaging with Jax and enhancing Bayex, your contributions would be highly welcomed and appreciated.

[**Installation**](#installation)
| [**Usage**](#usage)
| [**Reference docs**](https://bayex.readthedocs.io/en/latest/)

Bayex is a lightweight Bayesian optimization library designed for efficiency and flexibility, leveraging the power of JAX for high-performance numerical computations.
This library aims to provide an easy-to-use interface for optimizing expensive-to-evaluate functions through Gaussian Process (GP) models and various acquisition functions. Whether you're maximizing or minimizing your objective function, Bayex offers a simple yet powerful set of tools to guide your search for optimal parameters.

<p align="center">
    <img src="https://github.com/alonfnt/bayex/assets/38870744/882fecc7-bc30-4267-ad1d-687fdbbe2cdc" style="width: 720px; height: auto;">
</p>


## Installation
Bayex can be installed using [PyPI](https://pypi.org/project/bayex/) via `pip`:
```
pip install bayex
```

## Usage
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
ys = [f(x) for x in params['x']]
opt_state = optimizer.init(ys, params)

# Sample new points using Jax PRNG approach.
ori_key = jax.random.key(42)
for step in range(20):
    key = jax.random.fold_in(ori_key, step)
    new_params = optimizer.sample(key, opt_state)
    y_new = f(**new_params)
    opt_state = optimizer.fit(opt_state, y_new, new_params)
```

with the results being saved at `opt_state.best_params`.

## Documentation
Available at [https://bayex.readthedocs.io/en/latest](https://bayex.readthedocs.io/en/latest/).

## License
Bayex is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
