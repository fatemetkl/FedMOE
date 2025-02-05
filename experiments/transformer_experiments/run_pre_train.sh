#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=4G
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --job-name=5ett_pre_train
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --time=10:00:00



CONFIG_PATH=$1
VENV_PATH=$2

# Load the model output dir from the config file
MODEL_DIR=$(grep 'models_dir' $CONFIG_PATH | awk '{print $2}' | tr -d '"')
# Create the model output directory if it doesn't exist
mkdir -p $MODEL_DIR

SEED=2026


# Source the environment
source ${VENV_PATH}bin/activate
echo "Active Environment:"
which python

RUN_OUTPUT_FILE="${MODEL_DIR}log.out"

python -m experiments.transformer_experiments.pre_train_transformer \
    --config_path ${CONFIG_PATH} \
    --random_seed ${SEED} \
    > ${RUN_OUTPUT_FILE} 2>&1 &

    wait
    # Create a file that verifies that the Run concluded properly
    touch "${MODEL_DIR}done.out"
