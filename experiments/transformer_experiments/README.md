
`run_hp_sweep.sh` schedules each experiment through slurm with specific CPU allocation.

To run the pre-trained transformer experiment on cluster, first specify the experiment setup in the `config.yaml` file, then, set the hyper-parameter search space in the `run_hp_sweep.sh` script file. Note that these set of experiments first pre-train a transformer, then fine-tune the final layer in the main algorithm.
Run the experiment with the following command:

```
bash experiments/transformer_experiments/run_hp_sweep.sh \
path_to_config.yaml \
path_to_folder_for_artifacts/ \
path_to_desired_venv/
```

For example:

```
bash experiments/transformer_experiments/run_hp_sweep.sh \
experiments/transformer_experiments/config.yaml \
experiments/transformer_experiments/transformer_results/ \
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

For example, for the transformer experiment, if the output files are located at: `experiments/transformer_experiments/transformer_results/`, we should run:
Don't forget to complete the path to the experiment directory.
```
python -m experiments.find_best_hp --hp_sweep_dir experiments/transformer_experiments/transformer_results/
```
Make sure your environment is activated.