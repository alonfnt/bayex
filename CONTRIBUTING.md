# Contributing
In order to contribute code, one must begin by forking the repository. This creates a copy of the repository on your account.

Bayex uses poetry as a packaging and dependency manager. Hence, once you have cloned your repo on your own machine, you can use
```
poetry install
```
to install on the dependencies needed.
You can follow the instructions [here](https://python-poetry.org/docs/#installation) to setup your poetry installation.

You should start a new branch to write your changes on
```
git checkout -b name-of-change
```
or
```
git branch name-of-change
git checkout name-of-change
```

It is welcome if PR are composed of a single commit, to keep the feature <-> commit balance.
Please, when writing the commit message, try follow the [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) specitifcation.
Once you have made your changes and created your commit, it is recommended to run the pre-commit checks.
```
pre-commit run --all
```
as well as the tests to make sure everything works
```
pytest tests/
```

Remember to amend your current commit with the fixes if any of the checks fails.
