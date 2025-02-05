# Run on cluster
### Step 1: Transformer pre-training
Change the last argument to the path of your own virtual environment.

```
sbatch experiments/transformer_experiments/run_pre_train.sh \
experiments/transformer_experiments/boc_exchange/5_clients/pre_train_config.yaml \
experiments/transformer_experiments/boc_exchange/5_clients/models/ \
~/venv/fedmoe_env/

```

### Step 2: Run the main algorithm
Change the last argument to the path of your own virtual environment.

***GAME***
```
bash experiments/transformer_experiments/boc_exchange/game_run_hp_sweep.sh \
experiments/transformer_experiments/boc_exchange/5_clients/game_config.yaml \
experiments/transformer_experiments/boc_exchange/5_clients/results/ \
~/venv/fedmoe_env/
```
***Non-GAME***
```
bash experiments/transformer_experiments/boc_exchange/non_game_run_hp_sweep.sh \
experiments/transformer_experiments/boc_exchange/5_clients/non_game_config.yaml \
experiments/transformer_experiments/boc_exchange/5_clients/results/ \
~/venv/fedmoe_env/
```

### Step 3: Find the best hyper-parameters
Make sure your environment is activated.
Don't forget to complete the path to the experiment directory by changing the experiment name.
```
python -m experiments.find_best_hp --hp_sweep_dir experiments/transformer_experiments/boc_exchange/5_clients/results/experiment_name
```

# Run on your local machine

### Step 1: Activate your environment
First activate the environment.
For example:
```
source ../fedmoe_env/bin/activate

```
### Step 2: Pre-training
Then run the pre-training:

```
bash experiments/transformer_experiments/local_run_pre_train.sh \
experiments/transformer_experiments/boc_exchange/5_clients/pre_train_config.yaml
```


### Step 3: Main algorithm

```
bash experiments/transformer_experiments/boc_exchange/local_run_hp_sweep.sh \
experiments/transformer_experiments/boc_exchange/5_clients/config.yaml \
experiments/transformer_experiments/boc_exchange/5_clients/results/
```

### Step 4: Find the best hyper-parameters
Make sure your environment is activated. Don't forget to replace the `experiment_name` with your experiment name.
Find the best hyper-parameter:
```
python -m experiments.find_best_hp --hp_sweep_dir experiments/transformer_experiments/boc_exchange/5_clients/results/experiment_name
```
