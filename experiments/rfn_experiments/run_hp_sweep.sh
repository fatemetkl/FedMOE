#!/bin/bash

CONFIG_PATH=$1

# Load the experiment name from the config file
EXPERIMENT_NAME=$(grep 'experiment_name' $CONFIG_PATH | awk '{print $2}' | tr -d '"')
RESULTS_DIR="experiments/results/$EXPERIMENT_NAME"
# Create the results directory if it doesn't exist
mkdir -p $RESULTS_DIR

# Hyper-parameters
# ALPHA_VALUES=( 0.001 0.01 1.0 10)
# GAMMA_VALUES=( 0.01 1 )
# SIGMA_VALUES=( 0.001 0.01 1.0 10)
# HIDDENDIM_VALUES=( 5 10 20 )
# T_VALUES=( 5 10 20 )
# K_VALUES=( 1 2 3 )
# ETA_VALUES=( 1.0 2.0  )

ALPHA_VALUES=( 0.01 0.1 1.0 10.0)
GAMMA_VALUES=( 0.01 )
SIGMA_VALUES=( 0.01 )
HIDDENDIM_VALUES=( 5 10 )
T_VALUES=( 5 )
K_VALUES=( 3.0 )
# eta values shuold be int
ETA_VALUES=( 1 )


RUN_NAMES=( "Run1" "Run2" "Run3" )

for HIDDEN_DIM in "${HIDDENDIM_VALUES[@]}";
do
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
            for T_Value in "${T_VALUES[@]}";
            do
              EXPERIMENT_SETUP="T${T_Value}_alpha${ALPHA_VALUE}_gamma${GAMMA_VALUE}_sigma${SIGMA_VALUE}_DZ${HIDDEN_DIM}_K${K_VALUE}_ETA${ETA_VALUE}"
              EXPERIMENT_DIRECTORY="${RESULTS_DIR}/${EXPERIMENT_SETUP}/"
              mkdir -p $EXPERIMENT_DIRECTORY
              echo "Beginning Experiment ${EXPERIMENT_NAME} with hyper-parameters ${EXPERIMENT_SETUP}"
              for RUN_NAME in "${RUN_NAMES[@]}";
                do
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
                  --T ${T_Value} \
                  --experiment_setup ${EXPERIMENT_SETUP} \
                  > ${RUN_OUTPUT_FILE} 2>&1 &
              done
            done
          done
        done
      done
    done
  done
done
echo "Experiment $EXPERIMENT_NAME completed. Results are saved in $RESULTS_DIR."