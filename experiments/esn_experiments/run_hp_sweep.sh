#!/bin/bash

CONFIG_PATH=$1
ARTIFACTS_DIR=$2
VENV_PATH=$3

# Load the experiment name from the config file
EXPERIMENT_NAME=$(grep 'experiment_name' $CONFIG_PATH | awk '{print $2}' | tr -d '"')
RESULTS_DIR="${ARTIFACTS_DIR}${EXPERIMENT_NAME}"
# Create the results directory if it doesn't exist
mkdir -p $RESULTS_DIR


echo "CONFIG_PATH"${CONFIG_PATH}
echo "ARTIFACTS_DIR"${ARTIFACTS_DIR}



ALPHA_VALUES=( 0.1 1.0 10 20 )
GAMMA_VALUES=( 0.01 0.1 1.0 10 )
SIGMA_VALUES=( 0.01 1.0 )
HIDDENDIM_VALUES=( 2 5 10 20 30 40 )
T_VALUES=( 1 )
K_VALUES=( 1.0 2.0 )
ETA_VALUES=( 1.0 2.0 )

# ALPHA_VALUES=( 0.1 )
# GAMMA_VALUES=( 10 )
# SIGMA_VALUES=( 1.0 )
# HIDDENDIM_VALUES=( 2 )
# T_VALUES=( 1 )
# K_VALUES=( 1.0 )
# ETA_VALUES=( 2.0 )

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

                SBATCH_COMMAND="experiments/esn_experiments/run_fold_experiment.slrm \
                    ${CONFIG_PATH} \
                    ${EXPERIMENT_DIRECTORY} \
                    ${HIDDEN_DIM} \
                    ${ALPHA_VALUE} \
                    ${GAMMA_VALUE} \
                    ${SIGMA_VALUE} \
                    ${K_VALUE} \
                    ${ETA_VALUE} \
                    ${T_Value} \
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