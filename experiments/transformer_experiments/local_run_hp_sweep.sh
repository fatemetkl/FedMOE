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

ALPHA_VALUES=( 0.1 1 10)
GAMMA_VALUES=( 10 )
SIGMA_VALUES=( 0.01 )
# Hidden dimension should be divisible by nheads (4 by default)
HIDDENDIM_VALUES=( 4 )
T_VALUES=( 5 8 10 )
K_VALUES=( 1.0 )
ETA_VALUES=( 1.0 )
DATA_LOADER_NUM_SAMPLES=( 100 )
DATA_LOADER_BATCH_SIZE=( 10 )
PRE_TRAINING_EPOCHS=( 10 )
PRE_TRAINING_LEARNING_RATE=( 0.01 )



# RUN_NAMES=( "Run1" "Run2" "Run3" )
RUN_NAMES=( "Run1" )
SEEDS=(2024 2025 2026)
for HIDDEN_DIM in "${HIDDENDIM_VALUES[@]}"; do
  for ALPHA_VALUE in "${ALPHA_VALUES[@]}"; do
    for GAMMA_VALUE in "${GAMMA_VALUES[@]}"; do
      for SIGMA_VALUE in "${SIGMA_VALUES[@]}"; do
        for K_VALUE in "${K_VALUES[@]}"; do
          for ETA_VALUE in "${ETA_VALUES[@]}"; do
            for T_Value in "${T_VALUES[@]}"; do
              for NUM_SAMPLES in "${DATA_LOADER_NUM_SAMPLES[@]}"; do
                for BATCH_SIZE in "${DATA_LOADER_BATCH_SIZE[@]}"; do
                  for EPOCHS in "${PRE_TRAINING_EPOCHS[@]}"; do
                    for LEARNING_RATE in "${PRE_TRAINING_LEARNING_RATE[@]}"; do

                        EXPERIMENT_SETUP="T${T_Value}_alpha${ALPHA_VALUE}_gamma${GAMMA_VALUE}_sigma${SIGMA_VALUE}_DZ${HIDDEN_DIM}_K${K_VALUE}_ETA${ETA_VALUE}_pre_training${NUM_SAMPLES}-${BATCH_SIZE}-${EPOCHS}-${LEARNING_RATE}"
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
                                --T ${T_Value} \
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
  done
done
echo "Experiment $EXPERIMENT_NAME completed. Results are saved in $RESULTS_DIR."