
`local_run_hp_sweep.sh` runs the experiment locally, and `run_hp_sweep.sh` schedules each experiment through slurm with specific CPU allocation.

To run the random feature network experiment on cluster, first specify the experiment setup in the `config.yaml` file, then, set the hyper-parameter search space in the `run_hp_sweep.sh` script file. Then, run it with the following command:

```
bash experiments/rfn_experiments/run_hp_sweep.sh \
path_to_config.yaml \
path_to_folder_for_artifacts/ \
path_to_desired_venv/
```

For example:

```
bash experiments/rfn_experiments/run_hp_sweep.sh \
experiments/rfn_experiments/config.yaml \
experiments/rfn_experiments/rfn_results/ \
~/venv/fedmoe_env/
```

Important points:
- Even if you are not playing the game, don't set T to zero. Just set `have_sync: False` in `config.yaml`.


Results of the experiment including plots will be saved at: (EXPERIMENT_NAME is set in config file)
```
RESULTS_DIR="path_to_folder_for_artifacts/$EXPERIMENT_NAME"
```

To find the best hp based on the server loss, run 'find_best_hp.py' with the path to the experiment output file as the argument. We assume the last line of each log file contains the loss.

```
RESULTS_DIR="path_to_folder_for_artifacts/$EXPERIMENT_NAME"

poetry run python -m experiments.find_best_hp --hp_sweep_dir RESULTS_DIR
```

for example, for the RFN experiment, if the output files are located at: `experiments/rfn_experiments/rfn_results/`
So, we should run:


```
poetry run python -m experiments.find_best_hp --hp_sweep_dir experiments/rfn_experiments/rfn_results/
```
