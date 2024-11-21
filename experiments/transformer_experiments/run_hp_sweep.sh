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


ALPHA_VALUES=( 0.0005 0.001 0.005 0.01 )
GAMMA_VALUES=( 1 10 15 20 25 )
# Sigma is not used in transformer
SIGMA_VALUES=( 0.1 )
# Hidden dimension should be divisible by nheads (4 by default)
HIDDENDIM_VALUES=( 8 )
CLIENT_T_VALUES=( 5 10 15 20 )
# Remember to set this for the game and set to 0 for non-game settings.
GAME_SYNC_VALUES=( 2 3 4 5 )
K_VALUES=( 1.0 )
ETA_VALUES=( 1.0 )
DATA_LOADER_NUM_SAMPLES=( 200 )
DATA_LOADER_BATCH_SIZE=( 20 )
PRE_TRAINING_EPOCHS=( 200 )
PRE_TRAINING_LEARNING_RATE=( 0.01 )


# No game bests
# ALPHA_VALUES=( 0.1 )
# GAMMA_VALUES=( 10 )
# SIGMA_VALUES=( 0.1 )
# HIDDENDIM_VALUES=( 8 )
# CLIENT_T_VALUES=( 10 )
# GAME_SYNC_VALUES=( 0 )
# K_VALUES=( 1.0 )
# ETA_VALUES=( 1.0 )
# DATA_LOADER_NUM_SAMPLES=( 200 )
# DATA_LOADER_BATCH_SIZE=( 20 )
# PRE_TRAINING_EPOCHS=( 1 )
# PRE_TRAINING_LEARNING_RATE=( 0.01 )

# Game bests
# ALPHA_VALUES=( 0.005 )
# GAMMA_VALUES=( 20 )
# SIGMA_VALUES=( 0.1 )
# HIDDENDIM_VALUES=( 8 )
# CLIENT_T_VALUES=( 10 )
# GAME_SYNC_VALUES=( 3 )
# K_VALUES=( 1.0 )
# ETA_VALUES=( 1.0 )
# DATA_LOADER_NUM_SAMPLES=( 200 )
# DATA_LOADER_BATCH_SIZE=( 20 )
# PRE_TRAINING_EPOCHS=( 1 )
# PRE_TRAINING_LEARNING_RATE=( 0.01 )


for HIDDEN_DIM in "${HIDDENDIM_VALUES[@]}"; do
  for ALPHA_VALUE in "${ALPHA_VALUES[@]}"; do
    for GAMMA_VALUE in "${GAMMA_VALUES[@]}"; do
      for SIGMA_VALUE in "${SIGMA_VALUES[@]}"; do
        for K_VALUE in "${K_VALUES[@]}"; do
          for ETA_VALUE in "${ETA_VALUES[@]}"; do
            for Client_T_Value in "${CLIENT_T_VALUES[@]}"; do
              for GAME_SYNC in "${GAME_SYNC_VALUES[@]}"; do
                for NUM_SAMPLES in "${DATA_LOADER_NUM_SAMPLES[@]}"; do
                  for BATCH_SIZE in "${DATA_LOADER_BATCH_SIZE[@]}"; do
                    for EPOCHS in "${PRE_TRAINING_EPOCHS[@]}"; do
                      for LEARNING_RATE in "${PRE_TRAINING_LEARNING_RATE[@]}"; do

                          EXPERIMENT_SETUP="T${Client_T_Value}_sync${GAME_SYNC}_alpha${ALPHA_VALUE}_gamma${GAMMA_VALUE}_sigma${SIGMA_VALUE}_DZ${HIDDEN_DIM}_K${K_VALUE}_ETA${ETA_VALUE}_pre_training${NUM_SAMPLES}-${BATCH_SIZE}-${EPOCHS}-${LEARNING_RATE}"
                          EXPERIMENT_DIRECTORY="${RESULTS_DIR}/${EXPERIMENT_SETUP}/"
                          mkdir -p $EXPERIMENT_DIRECTORY
                          echo "Beginning Experiment ${EXPERIMENT_NAME} with hyper-parameters ${EXPERIMENT_SETUP}"

                          SBATCH_COMMAND="experiments/transformer_experiments/run_fold_experiment.slrm \
                              ${CONFIG_PATH} \
                              ${EXPERIMENT_DIRECTORY} \
                              ${HIDDEN_DIM} \
                              ${ALPHA_VALUE} \
                              ${GAMMA_VALUE} \
                              ${SIGMA_VALUE} \
                              ${K_VALUE} \
                              ${ETA_VALUE} \
                              ${Client_T_Value} \
                              ${GAME_SYNC} \
                              ${NUM_SAMPLES} \
                              ${BATCH_SIZE} \
                              ${EPOCHS} \
                              ${LEARNING_RATE} \
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
        done
      done
    done
  done
done
echo "Experiment $EXPERIMENT_NAME completed. Results are saved in $RESULTS_DIR."