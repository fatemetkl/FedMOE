#!/bin/bash

CONFIG_PATH=$1
ARTIFACTS_DIR=$2

# Load the experiment name from the config file
EXPERIMENT_NAME=$(grep 'experiment_name' $CONFIG_PATH | awk '{print $2}' | tr -d '"')
RESULTS_DIR="${ARTIFACTS_DIR}${EXPERIMENT_NAME}"
# Create the results directory if it doesn't exist
mkdir -p $RESULTS_DIR

# Hyper-parameters
ALPHA_VALUES=( 0.001 )
GAMMA_VALUES=( 10 )
SIGMA_VALUES=( 0.01 )
HIDDENDIM_VALUES=( 3 )
CLIENT_T_VALUES=( 5 )
# Game T values is the T used in equation 9.
GAME_T_VALUES=( 3 )
GAME_SYNC_VALUES=( 2 )
K_VALUES=( 3 )
ETA_VALUES=( 1 )


RUN_NAMES=( "Run1" "Run2" "Run3" )
SEEDS=( 2026 2025 2024 )
for HIDDEN_DIM in "${HIDDENDIM_VALUES[@]}"; do
  for ALPHA_VALUE in "${ALPHA_VALUES[@]}"; do
    for GAMMA_VALUE in "${GAMMA_VALUES[@]}"; do
      for SIGMA_VALUE in "${SIGMA_VALUES[@]}"; do
        for K_VALUE in "${K_VALUES[@]}"; do
          for ETA_VALUE in "${ETA_VALUES[@]}"; do
            for Client_T_Value in "${CLIENT_T_VALUES[@]}"; do
              for GAME_T_Value in "${GAME_T_VALUES[@]}"; do
                for game_sync in "${GAME_SYNC_VALUES[@]}"; do
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
                        python -m experiments.rfn_experiments.run_rfn_experiment \
                        --config_path ${CONFIG_PATH} \
                        --result_dir ${RUN_OUTPUT_DIR} \
                        --hidden_dim ${HIDDEN_DIM} \
                        --alpha ${ALPHA_VALUE} \
                        --gamma ${GAMMA_VALUE} \
                        --sigma ${SIGMA_VALUE} \
                        --K ${K_VALUE} \
                        --eta ${ETA_VALUE} \
                        --client_T ${Client_T_Value} \
                        --game_sync_freq ${game_sync} \
                        --game_T ${GAME_T_Value} \
                        --random_seed ${SEED} \
                        > ${RUN_OUTPUT_FILE} 2>&1 &

                        wait
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