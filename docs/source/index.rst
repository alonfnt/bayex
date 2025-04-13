Bayex
===========================================

Minimal Bayesian Optimization in JAX
------------------------------------

.. image:: https://github.com/alonfnt/bayex/actions/workflows/tests.yml/badge.svg
    :target: https://github.com/alonfnt/bayex/actions/workflows/tests.yml
    :alt: Test status

.. note::
   Bayex is currently a minimal, personally developed implementation that requires further development for broader application.
   If you're interested in engaging with JAX and enhancing Bayex, your contributions would be highly welcomed and appreciated.

.. raw:: html

    <p align="center">
        <img src="https://github.com/alonfnt/bayex/assets/38870744/882fecc7-bc30-4267-ad1d-687fdbbe2cdc">
    </p>

Bayex is a lightweight Bayesian optimization library designed for efficiency and flexibility, leveraging the power of JAX for high-performance numerical computations.

This library aims to provide an easy-to-use interface for optimizing expensive-to-evaluate functions through Gaussian Process (GP) models and various acquisition functions. Whether you're maximizing or minimizing your objective function, Bayex offers a simple yet powerful set of tools to guide your search for optimal parameters.

Installation
------------

Bayex can be installed using `PyPI <https://pypi.org/project/bayex/>`_ via ``pip``:

.. code-block:: bash

    pip install bayex

Usage
-----

Using Bayex is quite simple despite its low-level approach:

.. code-block:: python

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

With the results being saved at ``opt_state``.

Contributing
------------

We welcome contributions to Bayex! Whether it's adding new features, improving documentation, or reporting issues, please feel free to make a pull request or open an issue.

License
-------

Bayex is licensed under the MIT License. See the `LICENSE <https://github.com/alonfnt/bayex/blob/main/LICENSE>`_ file for more details.

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
