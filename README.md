# FedMOE
Codebase for federated mixture of experts.


## Developing

### Installing dependencies

The development environment can be set up using
[poetry](https://python-poetry.org/docs/#installation). You can initialize and manage poetry using a [virtualenv](https://pypi.org/project/virtualenv/) with the specific Python version. For this project we use python 3.10.14 . First create and activate your virtualenv, and then install poetry given the `pyproject.toml` file.

```bash
virtualenv "ENV_PATH"
source "ENV_PATH/bin/activate"
pip install --upgrade pip poetry
poetry install --with "dev"
```

Note that the with command is installing all libraries required for the full development workflow.


### Format checks and coding guidelines
To invoke pre-commit hooks, you can install the pre-commit hooks to be run locally. Activate your environment and run:

```bash
pre-commit install
```
To run the checks on all the files:

```bash
pre-commit run --all-files

```