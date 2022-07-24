import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="bojax",
    version="0.0.1",
    description="A lightweight Bayesian Optimization library in JAX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alonfnt/bojax",
    author="Albert Alonso",
    author_email="aalonso@proton.me",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="bayesian-optimization, jax, bayesian, automatic-differentiation, optimization",
    package_dir={"": "bojax"},
    packages=find_packages(where="bojax"),
    python_requires=">=3.7, <4",
    install_requires=["jax, jaxlib"],
    extras_require={
        "dev": ["pytest", "pytest-cov", "pre-commit", "flake8", "mypy", "isort"],
        "test": ["coverage"],
    },
)
