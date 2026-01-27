
`run_hp_sweep.sh` schedules each experiment through slurm with specific CPU allocation.
Firstly, make sure to have your poetry environment activated (this is done by default when you activate your virtualenv). These scripts will use your poetry environment to run python codes.
To run the random Echo State experiment on cluster, first specify the experiment setup in the `config.yaml` file, then, set the hyper-parameter search space in the `run_hp_sweep.sh` script file. Then, run it with the following command:

```
bash experiments/esn_experiments/run_hp_sweep.sh \
path_to_config.yaml \
path_to_folder_for_artifacts/ \
path_to_desired_venv/
```
**The exact running commands are mentioned in the readme file of each experiment/dataset directory.**

Results of the experiment including plots will be saved at: (EXPERIMENT_NAME is set in config file)
```
RESULTS_DIR="path_to_folder_for_artifacts/$EXPERIMENT_NAME"
```

### Find the best hyper-parameters

To find the best hp based on the server loss, run 'find_best_hp.py' with the path to the experiment output file as the argument. We assume the last line of each log file contains the loss.

for example, for the ESN experiment, if the output files are located at: `experiments/esn_experiments/esn_results/`
So, we should run:

```
RESULTS_DIR="path_to_folder_for_artifacts/$EXPERIMENT_NAME"

python -m experiments.find_best_hp --hp_sweep_dir RESULTS_DIR
```

For example:

```
python -m experiments.find_best_hp --hp_sweep_dir experiments/esn_experiments/esn_results/
```

**The exact running commands are mentioned in the readme file of each experiment/dataset directory.**

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
bash experiments/esn_experiments/local_run_hp_sweep.sh \
path_to_config.yaml \
path_to_folder_for_artifacts/ \
```

### Find the best hyper-parameters

Find the best hyper-parameter:
```
python -m experiments.find_best_hp --hp_sweep_dir experiments/esn_experiments/esn_results/
```
Make sure your environment is activated.

**The exact running commands are mentioned in the readme file of each experiment/dataset directory.**
