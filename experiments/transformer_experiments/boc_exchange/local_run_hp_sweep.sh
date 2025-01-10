#!/bin/bash

CONFIG_PATH=$1
ARTIFACTS_DIR=$2

# Load the experiment name from the config file
EXPERIMENT_NAME=$(grep 'experiment_name' $CONFIG_PATH | awk '{print $2}' | tr -d '"')
RESULTS_DIR="${ARTIFACTS_DIR}${EXPERIMENT_NAME}"
# Create the results directory if it doesn't exist
mkdir -p $RESULTS_DIR

echo "CONFIG_PATH"${CONFIG_PATH}
echo "ARTIFACTS_DIR"${ARTIFACTS_DIR}

# Game bests
ALPHA_VALUES=( 0.002)
GAMMA_VALUES=( 20 )
SIGMA_VALUES=( 0.1 )
HIDDENDIM_VALUES=( 8 )
# Client T value is the T used in individual client optimization (equation 4).
CLIENT_T_VALUES=( 5 )
# Game T value is the T used in equation 9.
GAME_T_VALUES=( 3 )
# Game synchronization value is the frequency at which the game is played.
GAME_SYNC_VALUES=( 1 )
K_VALUES=( 1.0 )
ETA_VALUES=( 1.0 )
# If you are using pre-trained transformers, the below values do not matter.
DATA_LOADER_NUM_SAMPLES=( 200 )
DATA_LOADER_BATCH_SIZE=( 20 )
PRE_TRAINING_EPOCHS=( 1 )
PRE_TRAINING_LEARNING_RATE=( 0.01 )

# We don't need to run the transformer model several times as there is no randomness after the model is pre-trained.
# RUN_NAMES=( "Run1" "Run2" "Run3" )
RUN_NAMES=( "Run1" )
SEEDS=( 2026 )
for HIDDEN_DIM in "${HIDDENDIM_VALUES[@]}"; do
  for ALPHA_VALUE in "${ALPHA_VALUES[@]}"; do
    for GAMMA_VALUE in "${GAMMA_VALUES[@]}"; do
      for SIGMA_VALUE in "${SIGMA_VALUES[@]}"; do
        for K_VALUE in "${K_VALUES[@]}"; do
          for ETA_VALUE in "${ETA_VALUES[@]}"; do
            for CLIENT_T_VALUE in "${CLIENT_T_VALUES[@]}"; do
              for GAME_T_VALUE in "${GAME_T_VALUES[@]}"; do
                for GAME_SYNC in "${GAME_SYNC_VALUES[@]}"; do
                  EXPERIMENT_SETUP="T${Client_T_Value}_sync${game_sync}_gameT${GAME_T_Value}_alpha${ALPHA_VALUE}_gamma${GAMMA_VALUE}_sigma${SIGMA_VALUE}_DZ${HIDDEN_DIM}"
                  EXPERIMENT_DIRECTORY="${RESULTS_DIR}/${EXPERIMENT_SETUP}/"
                  mkdir -p $EXPERIMENT_DIRECTORY
                  echo "Beginning Experiment ${EXPERIMENT_NAME} with hyper-parameters ${EXPERIMENT_SETUP}"

                  for ((i=0; i<${#RUN_NAMES[@]}; i++));
                    do
                      RUN_NAME="${RUN_NAMES[i]}"
                      SEED="${SEEDS[i]}"
                      RUN_OUTPUT_DIR="${EXPERIMENT_DIRECTORY}${RUN_NAME}/"
                      RUN_OUTPUT_FILE="${RUN_OUTPUT_DIR}log.out"
                      mkdir "${RUN_OUTPUT_DIR}"

                      python -m experiments.transformer_experiments.run_transformer_experiment \
                          --config_path ${CONFIG_PATH} \
                          --result_dir ${RUN_OUTPUT_DIR} \
                          --hidden_dim ${HIDDEN_DIM} \
                          --alpha ${ALPHA_VALUE} \
                          --gamma ${GAMMA_VALUE} \
                          --sigma ${SIGMA_VALUE} \
                          --K ${K_VALUE} \
                          --eta ${ETA_VALUE} \
                          --client_T ${CLIENT_T_VALUE} \
                          --game_sync_freq ${GAME_SYNC} \
                          --game_T ${GAME_T_VALUE} \
                          --data_loader_num_samples ${DATA_LOADER_NUM_SAMPLES} \
                          --data_loader_batch_size ${DATA_LOADER_BATCH_SIZE} \
                          --pre_training_epochs ${PRE_TRAINING_EPOCHS} \
                          --pre_training_learning_rate ${PRE_TRAINING_LEARNING_RATE} \
                          --random_seed ${SEED} \
                          > ${RUN_OUTPUT_FILE} 2>&1 &

                          wait
                          # Create a file that verifies that the Run concluded properly
                          touch "${RUN_OUTPUT_DIR}done.out"
                        done
                      # wait 2 seconds before running the next command
                    sleep 2
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