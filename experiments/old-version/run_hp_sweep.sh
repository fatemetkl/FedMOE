#!/bin/bash

CONFIG_PATH=$1

# Load the experiment name from the config file
EXPERIMENT_NAME=$(grep 'experiment_name' $CONFIG_PATH | awk '{print $2}' | tr -d '"')
RESULTS_DIR="results/$EXPERIMENT_NAME"
# Create the results directory if it doesn't exist
mkdir -p $RESULTS_DIR

# Hyper-parameters
ALPHA_VALUES=( 0.0001 0.001 0.01 0.1 1.0 )
GAMMA_VALUES=( 0.00001 0.0001 0.001 0.01 0.1 )
SIGMA_VALUES=( 0.00001 0.0001 0.001 0.01 0.1 )
K_VALUES=( 1 2 3 )
ETA_VALUES=( 1.0 2.0  )


for ALPHA_VALUE in "${ALPHA_VALUES[@]}";
do
  for GAMMA_VALUE in "${GAMMA_VALUES[@]}";
  do
    for SIGMA_VALUE in "${SIGMA_VALUES[@]}";
    do
        for K_VALUE in "${K_VALUES[@]}";
        do
            for ETA_VALUE in "${ETA_VALUES[@]}";
            do
                EXPERIMENT_SETUP="alpha_${ALPHA_VALUE}_gamma_${GAMMA_VALUE}_sigma_${SIGMA_VALUE}_K${K_VALUE}_ETA${ETA_VALUE}"
                echo "Beginning Experiment ${EXPERIMENT_NAME} with hyper-parameters ${EXPERIMENT_SETUP}"
                EXPERIMENT_DIRECTORY="${RESULTS_DIR}/${EXPERIMENT_SETUP}/"
                python -m experiments.run_experiment \
                --config_path ${CONFIG_PATH} \
                --alpha ${ALPHA_VALUE} \
                --gamma ${GAMMA_VALUE} \
                --sigma ${SIGMA_VALUE} \
                --K ${K_VALUE} \
                --ETA ${ETA_VALUE} \
                --result_dir ${EXPERIMENT_DIRECTORY} \
                --experiment_setup ${EXPERIMENT_SETUP} \

echo "Experiment $EXPERIMENT_NAME completed. Results are saved in $RESULTS_DIR."