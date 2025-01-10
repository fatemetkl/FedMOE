#!/bin/bash

CONFIG_PATH=$1
# Load the model output dir from the config file
MODEL_DIR=$(grep 'models_dir' $CONFIG_PATH | awk '{print $2}' | tr -d '"')
# Create the model output directory if it doesn't exist
mkdir -p $MODEL_DIR

SEED=2026


RUN_OUTPUT_FILE="${MODEL_DIR}log.out"

python -m experiments.transformer_experiments.pre_train_transformer \
    --config_path ${CONFIG_PATH} \
    --random_seed ${SEED} \
    > ${RUN_OUTPUT_FILE} 2>&1 &

    wait
    # Create a file that verifies that the Run concluded properly
    touch "${MODEL_DIR}done.out"
