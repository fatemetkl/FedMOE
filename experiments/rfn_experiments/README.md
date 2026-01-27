
`local_run_hp_sweep.sh` runs the experiment locally, and `run_hp_sweep.sh` schedules each experiment through slurm with specific CPU allocation.

**Note that for RFN experiments we only work with datasets with dy=1 since the game implementations are currently only implemented for dy=1.** Therefore, we have not experimented with the *concept_drift* and *brownian_addition* dataset here.

# Run on cluster
### Run the main experiments

To run the random feature network experiment on cluster, first specify the experiment setup in the `config.yaml` file, then, set the hyper-parameter search space in the `run_hp_sweep.sh` script file. Then, run it with the following command:

```
bash experiments/rfn_experiments/run_hp_sweep.sh \
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

```
RESULTS_DIR="path_to_folder_for_artifacts/$EXPERIMENT_NAME"

python -m experiments.find_best_hp --hp_sweep_dir RESULTS_DIR
```

for example, for the RFN experiment, if the output files are located at: `experiments/rfn_experiments/rfn_results/`
So, we should run:


```
python -m experiments.find_best_hp --hp_sweep_dir experiments/rfn_experiments/rfn_results/
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
bash experiments/rfn_experiments/local_run_hp_sweep.sh \
path_to_config.yaml \
path_to_folder_for_artifacts/ \
```


### Find the best hyper-parameters
Find the best hyper-parameter:
```
python -m experiments.find_best_hp --hp_sweep_dir experiments/rfn_experiments/rfn_results/
```
Make sure your environment is activated.


**The exact running commands are mentioned in the readme file of each experiment/dataset directory.**
