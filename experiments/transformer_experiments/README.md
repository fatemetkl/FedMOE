
`run_hp_sweep.sh` schedules each experiment through slurm with specific CPU allocation.

To run the pre-trained transformer experiment on cluster, first specify the experiment setup in the `config.yaml` file, then, set the hyper-parameter search space in the `run_hp_sweep.sh` script file. Note that these set of experiments first pre-train a transformer, then fine-tune the final layer in the main algorithm.
Run the experiment with the following command:

First you need to pre-train transformers for each client for each dataset. Specify name of data and settings in the pre_train_config.yaml file.
Run:
```
sbatch experiments/transformer_experiments/run_pre_train.slrm
```
Then you can use the same pre-trained transformers across different hyper-parameters. In this case, the hyper-parameter sweep should
not consider hidden dimension because it is fixed in the pre_trained model.
To launch the main algorithm make sure to set the pre_training epochs to zero if you want to use saved models, then run:

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



## Run hp sweep on you local machine

First activate the environment.
For example:
```
source ../fedmoe_env/bin/activate

```
#### Pre-train the transformer models
For each client, we train a transformer and fix it across different hyper-parameter settings.
Make sure to create RUN_OUTPUT_FILE yourself.

```
RUN_OUTPUT_FILE="${RUN_OUTPUT_DIR}log.out"

python -m experiments.transformer_experiments.pre_train_transformer \
    --config_path ${CONFIG_PATH} \
    --random_seed ${SEED} \
    > ${RUN_OUTPUT_FILE} 2>&1 &
```

For example:

```
python -m experiments.transformer_experiments.pre_train_transformer \
    --config_path "experiments/transformer_experiments/pre_train_config.yaml" \
    --random_seed 2026 \
    > "experiments/transformer_experiments/models/log.out" 2>&1 &

```

#### Run experiments
To launch the experiments run:

```
bash experiments/transformer_experiments/local_run_hp_sweep.sh \
path_to_config.yaml \
path_to_folder_for_artifacts/ \
```

For example:

```
bash experiments/transformer_experiments/local_run_hp_sweep.sh \
experiments/transformer_experiments/config.yaml \
experiments/transformer_experiments/transformer_results/
```

Find the best hyper-parameter:
```
python -m experiments.find_best_hp --hp_sweep_dir experiments/transformer_experiments/transformer_results/
```
Make sure your environment is activated.