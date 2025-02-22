
`local_run_hp_sweep.sh` runs the experiment locally, and `run_hp_sweep.sh` schedules each experiment through slurm with specific CPU allocations.
# Run on cluster
### Run the main experiments

To run the random feature network experiment on cluster, first specify the experiment setup in the `config.yaml` file, then, set the hyper-parameter search space in the `run_hp_sweep.sh` script file. Then, run it with the following command.
Change the last argument to the path of your own virtual environment.

**GAME**

```
bash experiments/rfn_experiments/covariate_shift/game_run_hp_sweep.sh \
experiments/rfn_experiments/covariate_shift/game_config.yaml \
experiments/rfn_experiments/covariate_shift/results/ \
~/venv/fedmoe_env/
```
**Non GAME**

```
bash experiments/rfn_experiments/covariate_shift/non_game_run_hp_sweep.sh \
experiments/rfn_experiments/covariate_shift/non_game_config.yaml \
experiments/rfn_experiments/covariate_shift/results/ \
~/venv/fedmoe_env/
```

Results of the experiment including plots will be saved at: (EXPERIMENT_NAME is set in config file)
```
RESULTS_DIR="experiments/rfn_experiments/covariate_shift/results/$EXPERIMENT_NAME"
```
### Find the best hyper-parameters
To find the best hp based on the server loss, run 'find_best_hp.py' with the path to the experiment output file as the argument. We assume the last line of each log file contains the loss.

Don't forget to complete the path to the experiment directory by changing the experiment name.
```
python -m experiments.find_best_hp --hp_sweep_dir experiments/rfn_experiments/covariate_shift/results/experiment_name
```

### Run the experiment with the best hyper-parameters
Make sure to set visualization variables to `True` in `config.yaml`. Set the best hyper-parameters in `game_best_hp.sh` and `non_game_best_hp.sh` files.

*** With GAME ***
```
bash experiments/rfn_experiments/covariate_shift/game_best_hp.sh \
experiments/rfn_experiments/covariate_shift/game_config.yaml \
experiments/rfn_experiments/covariate_shift/best_results/ \
~/venv/fedmoe_env/

```

*** No GAME ***

```
bash experiments/rfn_experiments/covariate_shift/non_game_best_hp.sh \
experiments/rfn_experiments/covariate_shift/non_game_config.yaml \
experiments/rfn_experiments/covariate_shift/best_results/ \
~/venv/fedmoe_env/
```


# Run on your local machine

### Activate your environment
First, activate the environment.
For example:
```
source ../fedmoe_env/bin/activate

```
#### Run experiments
To launch the experiments run:

```
bash experiments/rfn_experiments/covariate_shift/local_run_hp_sweep.sh \
experiments/rfn_experiments/covariate_shift/config.yaml \
experiments/rfn_experiments/covariate_shift/results/
```

### Find the best hyper-parameters
Make sure your environment is activated.
Don't forget to replace the `experiment_name` with your experiment name. Find the best hyper-parameter:

```
python -m experiments.find_best_hp --hp_sweep_dir experiments/rfn_experiments/covariate_shift/results/experiment_name
```
