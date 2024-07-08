
To run the random feature network experiment, first specify the experiment seetup in the `config.yaml` file, then, set the hyper-parameter search space in the `run_hp_sweep.sh` script file. Then, run it with the following command:

```
bash experiments/rfn_experiments/run_hp_sweep.sh experiments/rfn_experiments/config.yaml
```

The output of this command is the set of optimal hyper parameters based on the specified metric in `run_rfn_experiment.py` file alongside the final metric value.



To find the best hp based on the server loss, run 'find_best_hp.py' with the path to the experiment output file as the argument. We assume the last line of each log file contains the loss.

```
RESULTS_DIR="experiments/results/$EXPERIMENT_NAME"
```
for example, for the RFN experiment, the output files are located at: `experiments/results/rfn-experiment/`
So, we should run:

```
python -m experiments.find_best_hp --hp_sweep_dir experiments/results/rfn-experiment/
```
