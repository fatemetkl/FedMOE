# FedMOE
Codebase for federated mixture of experts.

## Codebase Structure

The main code of the algorithm is located in `fedmoe`. You can re-generate experiment results and artifacts by running the code in `experiments`.

### Run Experiments
The experimental scripts are all located in the `experiments` directory. Three main models are considered: `Echo State Network (ESN)`, `Random Feature Network (RFN)`, and `Pre-trained Transformer` model. You can find the experimental scripts for each dataset under the model experiment directory. Provided README files mention the exact command needed to run each experiment. All experiments include scripts for hyper-parameter tuning. Don't forget to install and activate your virtual environment as described [here](#developing).
#### Example Running Script
Note that slurm scrips to run the codes on a HPC environment are also provided. Here is an example of running one of the experiments on your local machine after if you already know the value of hyper-parameters. This experiment runs ESN experts, and the server. Details such as number of clients and the dataset name are specified in `config.yaml`, and its address should be specified as `CONFIG_PATH`.

```bash
python -m experiments.esn_experiments.run_esn_experiment \
        --config_path ${CONFIG_PATH} \
        --result_dir ${RUN_OUTPUT_DIR} \
        --hidden_dim ${HIDDEN_DIM} \
        --alpha ${ALPHA_VALUE} \
        --gamma ${GAMMA_VALUE} \
        --sigma ${SIGMA_VALUE} \
        --K ${K_VALUE} \
        --eta ${ETA_VALUE} \
        --client_T ${CLIENT_T_VALUE} \
        --game_sync_freq ${GAME_SYNC} \
        --game_T ${GAME_T_VALUE} \
        --random_seed ${SEED} \
        > ${RUN_OUTPUT_FILE} 2>&1 &
```

### `fedmoe` Components

`client_manager` manages a group of clients or experts and acts as a medium between the clients and the server.
`server` implements the main time loop of the algorithm as described in the paper (See Algorithm 2 in the Appendix).

- [datasets](https://github.com/fatemetkl/FedMOE/tree/main/fedmoe/datasets): Contains scripts related to generating, loading, processing, and managing the datasets.
- [clients](https://github.com/fatemetkl/FedMOE/tree/main/fedmoe/clients): Contains implementations of various client types, each with specific models and functionalities. These clients represent the participants or experts in our setup.
- [models](https://github.com/fatemetkl/FedMOE/tree/main/fedmoe/models): Includes different model architectures used in the experiments. Each model is designed to integrate with the FedMOE framework and can be tailored for specific tasks.
- [game](https://github.com/fatemetkl/FedMOE/tree/main/fedmoe/game): Implements the core logic of the Nash game, including expectation estimations and computation of matrices.
- [state](https://github.com/fatemetkl/FedMOE/tree/main/fedmoe/state): The state of the clients and the game is saved and managed throughout the time-series algorithm.
- [tests](https://github.com/fatemetkl/FedMOE/tree/main/fedmoe/tests): Contains unit tests to ensure the reliability and correctness of the code. These tests cover various components to prevent bugs and maintain code quality.
- [utils](https://github.com/fatemetkl/FedMOE/tree/main/fedmoe/utils): Scripts for various tasks, including data preprocessing, visualization of plots using the experiment artifacts (i.e., JSON files), and other reusable code snippets.

## Developing

### Installing dependencies

The development environment can be set up using [poetry](https://python-poetry.org/docs/#installation). You can initialize and manage poetry using a [virtualenv](https://pypi.org/project/virtualenv/) with a specific Python version. For this project, we use `python = ">=3.10.0,<3.11"`. First, create and activate your virtualenv, and then install poetry with the `pyproject.toml` file.

```bash
virtualenv "ENV_PATH"
source "ENV_PATH/bin/activate"
pip install --upgrade pip poetry
poetry install --with "dev"
```

Note that this command will install all libraries required for the full development workflow.

### Format checks and coding guidelines
To invoke pre-commit hooks, you can install the pre-commit hooks to be run locally. Activate your environment and run:

```bash
pre-commit install
```
To run the checks on all the files:

```bash
pre-commit run --all-files
```

## Citation
If you used our work (code or the paper) in your project or research, please use the citation below.
