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



ALPHA_VALUES=( 0.0001 0.001 0.01 )
GAMMA_VALUES=( 10 15 )
SIGMA_VALUES=( 0.01 0.1 )
HIDDENDIM_VALUES=( 3 4 5 )
# Client T value is the T used in individual client optimization (equation 4).
CLIENT_T_VALUES=( 3 4 5 )
# Remember to set this for the game and set to 0 for non-game settings.
# Game T value is the T used in equation 9.
GAME_T_VALUES=( 0 )
# Game synchronization value is the frequency at which the game is played.
GAME_SYNC_VALUES=( 0 )
K_VALUES=( 1.0 )
ETA_VALUES=( 1.0 )


for HIDDEN_DIM in "${HIDDENDIM_VALUES[@]}"; do
  for ALPHA_VALUE in "${ALPHA_VALUES[@]}"; do
    for GAMMA_VALUE in "${GAMMA_VALUES[@]}"; do
      for SIGMA_VALUE in "${SIGMA_VALUES[@]}"; do
        for K_VALUE in "${K_VALUES[@]}"; do
          for ETA_VALUE in "${ETA_VALUES[@]}"; do
            for CLIENT_T_VALUE in "${CLIENT_T_VALUES[@]}"; do
              for GAME_T_VALUE in "${GAME_T_VALUES[@]}"; do
                for GAME_SYNC in "${GAME_SYNC_VALUES[@]}"; do
                    EXPERIMENT_SETUP="T${CLIENT_T_VALUE}_sync${GAME_SYNC}_gameT${GAME_T_VALUE}_alpha${ALPHA_VALUE}_gamma${GAMMA_VALUE}_sigma${SIGMA_VALUE}_DZ${HIDDEN_DIM}"
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
                        ${CLIENT_T_VALUE} \
                        ${GAME_SYNC} \
                        ${GAME_T_VALUE}\
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
echo "Experiment $EXPERIMENT_NAME completed. Results are saved in $RESULTS_DIR."