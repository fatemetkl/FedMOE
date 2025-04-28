`local_run_hp_sweep.sh` runs the experiment locally, and `run_hp_sweep.sh` schedules each experiment through slurm with specific CPU allocation.

# Run on cluster
### Run the main experiments

To run the random Echo State experiment on cluster, first specify the experiment setup in the `config.yaml` file, then, set the hyper-parameter search space in the `run_hp_sweep.sh` script file. Then, run it with the following command:
Change the last argument to the path of your own virtual environment.

**Experiment with the game**:
Config is specific to game.
```
bash experiments/esn_experiments/boc_exchange/game_run_hp_sweep.sh \
experiments/esn_experiments/boc_exchange/game_config.yaml \
experiments/esn_experiments/boc_exchange/results/ \
~/venv/fedmoe_env/
```

**Experiment with NO game**:
Config is specific to non-game.
```
bash experiments/esn_experiments/boc_exchange/non_game_run_hp_sweep.sh \
experiments/esn_experiments/boc_exchange/non_game_config.yaml \
experiments/esn_experiments/boc_exchange/results/ \
~/venv/fedmoe_env/
```

Results of the experiment including plots will be saved at: (EXPERIMENT_NAME is set in config file)
```
RESULTS_DIR="path_to_folder_for_artifacts/$EXPERIMENT_NAME"
```
### Find the best hyper-parameters
To find the best hp based on the server loss, run `find_best_hp.py` with the path to the experiment output file as the argument. We assume the last line of each log file contains the loss.

Don't forget to complete the path to the experiment directory by changing the experiment name.

```
python -m experiments.find_best_hp --hp_sweep_dir experiments/esn_experiments/boc_exchange/results/experiment_name
```

### Run the experiment with the best hyper-parameters
Make sure to set visualization variables to `True` in `config.yaml`.

*** With GAME ***
```
bash experiments/esn_experiments/boc_exchange/game_two_clients_best_hp.sh \
experiments/esn_experiments/boc_exchange/game_config.yaml \
experiments/esn_experiments/boc_exchange/best_results/ \
~/venv/fedmoe_env/
```
*** No GAME ***

```
bash experiments/esn_experiments/boc_exchange/non_game_two_clients_best_hp.sh \
experiments/esn_experiments/boc_exchange/non_game_config.yaml \
experiments/esn_experiments/boc_exchange/best_results/ \
~/venv/fedmoe_env/
```

# Run on your local machine
### Activate your environment

First activate the environment.
For example:
```
source ../fedmoe_env/bin/activate

```

#### Run experiments
To launch the experiments run:

```
bash experiments/esn_experiments/boc_exchange/local_run_hp_sweep.sh \
experiments/esn_experiments/boc_exchange/config.yaml \
experiments/esn_experiments/boc_exchange/results/
```

### Find the best hyper-parameters
Make sure your environment is activated. Don't forget to replace the `experiment_name` with your experiment name. Find the best hyper-parameter:

```
python -m experiments.find_best_hp --hp_sweep_dir experiments/esn_experiments/boc_exchange/results/experiment_name
```
