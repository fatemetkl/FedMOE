`local_run_hp_sweep.sh` runs the experiment locally, and `run_hp_sweep.sh` schedules each experiment through slurm with specific CPU allocation.

# Run on cluster
### Run the main experiments

To run the random Echo State experiment on cluster, first specify the experiment setup in the `config.yaml` file, then, set the hyper-parameter search space in the `run_hp_sweep.sh` script file. Then, run it with the following command:
Change the last argument to the path of your own virtual environment.

**Experiment with the game**:
Config is specific to game.
```
bash experiments/esn_experiments/boc_exchange_validation/game_run_hp_sweep.sh \
experiments/esn_experiments/boc_exchange_validation/game_config.yaml \
experiments/esn_experiments/boc_exchange_validation/results/ \
~/venv/fedmoe_env/
```

**Experiment with NO game**:
Config is specific to non-game.
```
bash experiments/esn_experiments/boc_exchange_validation/non_game_run_hp_sweep.sh \
experiments/esn_experiments/boc_exchange_validation/non_game_config.yaml \
experiments/esn_experiments/boc_exchange_validation/results/ \
~/venv/fedmoe_env/
```

Results of the experiment including plots will be saved at: (EXPERIMENT_NAME is set in config file)
```
RESULTS_DIR="path_to_folder_for_artifacts/$EXPERIMENT_NAME"
```
