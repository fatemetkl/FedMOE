#!/bin/bash

CONFIG_PATH=$1

# Load the experiment name from the config file
EXPERIMENT_NAME=$(grep 'experiment_name' $CONFIG_PATH | awk '{print $2}' | tr -d '"')
RESULTS_DIR="results/$EXPERIMENT_NAME"
# Create the results directory if it doesn't exist
mkdir -p $RESULTS_DIR

python -m experiments.run_experiment --config_path ${CONFIG_PATH} --result_dir ${RESULTS_DIR}

echo "Experiment $EXPERIMENT_NAME completed. Results are saved in $RESULTS_DIR."