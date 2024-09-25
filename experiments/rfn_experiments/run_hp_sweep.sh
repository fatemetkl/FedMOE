#!/bin/bash

CONFIG_PATH=$1
VENV_PATH=$2

# Load the experiment name from the config file
EXPERIMENT_NAME=$(grep 'experiment_name' $CONFIG_PATH | awk '{print $2}' | tr -d '"')
RESULTS_DIR="experiments/results/$EXPERIMENT_NAME"
# Create the results directory if it doesn't exist
mkdir -p $RESULTS_DIR

# Hyper-parameters
# ALPHA_VALUES=( 0.001 0.01 1.0 10)
# GAMMA_VALUES=( 0.01 1 )
# SIGMA_VALUES=( 0.01 )
# HIDDENDIM_VALUES=( 5 10 )
# T_VALUES=( 5 10 )
# K_VALUES=( 3 )
# ETA_VALUES=( 1 )

ALPHA_VALUES=( 1.0 )
GAMMA_VALUES=( 1.0 )
SIGMA_VALUES=( 0.01 )
HIDDENDIM_VALUES=( 10 )
T_VALUES=( 8 )
K_VALUES=( 3.0 )
# eta values shuold be int
ETA_VALUES=( 2 )


for HIDDEN_DIM in "${HIDDENDIM_VALUES[@]}"; do
  for ALPHA_VALUE in "${ALPHA_VALUES[@]}"; do
    for GAMMA_VALUE in "${GAMMA_VALUES[@]}"; do
      for SIGMA_VALUE in "${SIGMA_VALUES[@]}"; do
        for K_VALUE in "${K_VALUES[@]}"; do
          for ETA_VALUE in "${ETA_VALUES[@]}"; do
            for T_Value in "${T_VALUES[@]}"; do

                EXPERIMENT_SETUP="T${T_Value}_alpha${ALPHA_VALUE}_gamma${GAMMA_VALUE}_sigma${SIGMA_VALUE}_DZ${HIDDEN_DIM}_K${K_VALUE}_ETA${ETA_VALUE}"
                EXPERIMENT_DIRECTORY="${RESULTS_DIR}/${EXPERIMENT_SETUP}/"
                mkdir -p $EXPERIMENT_DIRECTORY
                echo "Beginning Experiment ${EXPERIMENT_NAME} with hyper-parameters ${EXPERIMENT_SETUP}"

                SBATCH_COMMAND="experiments/rfn_experiments/run_fold_experiment.slrm \
                    ${CONFIG_PATH} \
                    ${EXPERIMENT_DIRECTORY} \
                    ${HIDDEN_DIM} \
                    ${ALPHA_VALUE} \
                    ${GAMMA_VALUE} \
                    ${SIGMA_VALUE} \
                    ${K_VALUE} \
                    ${ETA_VALUE} \
                    ${T_Value} \
                    ${EXPERIMENT_SETUP} \
                    ${VENV_PATH}"
                echo "Running sbatch command ${SBATCH_COMMAND}"
                sbatch ${SBATCH_COMMAND}
            done
          done
        done
      done
    done
  done
done
echo "Experiment $EXPERIMENT_NAME completed. Results are saved in $RESULTS_DIR."