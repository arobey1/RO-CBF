#!/bin/bash

export SYSTEM='carla'
export DATA_PATH_1='./data/carla/Data_Collection_Compiled_noisy2.pd'

export RESULTS_PATH='./results'

# Margins for optimization constraints 
export GAMMA_SAFE=0.3
export GAMMA_UNSAFE=0.3
export GAMMA_DYN=0.05

# Lagrange multipliers (fixed)
export LAMBDA_GRAD=0.01
export LAMBDA_PARAM=0.01

# Robustness
export LAMBDA_ROBUST=0.6

# Training
export NET_DIMS=(32 16)
export N_EPOCHS=10000
export LEARNING_RATE=0.005
export DUAL_STEP_SIZE=0.05
export DUAL_SCHEME='avg'

# Larger thresh --> more neighbors --> fewer boundary points
# Smaller thresh --> fewer neighbors --> more boundary points

# Boundary/Unsafe state sampling
export NEIGHBOR_THRESH=0.045
export MIN_N_NEIGHBORS=200

# Additional state sampling
export N_SAMP_UNSAFE=0
export N_SAMP_SAFE=0
export N_SAMP_ALL=0

# $DATA_PATH_2 $DATA_PATH_3

python main.py \
    --system $SYSTEM --data-path $DATA_PATH_1 --results-path $RESULTS_PATH \
    --gamma-safe $GAMMA_SAFE --gamma-unsafe $GAMMA_UNSAFE --gamma-dyn $GAMMA_DYN \
    --lambda-grad $LAMBDA_GRAD --lambda-param $LAMBDA_PARAM \
    --net-dims ${NET_DIMS[@]} --n-epochs $N_EPOCHS \
    --learning-rate $LEARNING_RATE --dual-step-size $DUAL_STEP_SIZE \
    --nbr-thresh $NEIGHBOR_THRESH --min-n-nbrs $MIN_N_NEIGHBORS \
    --n-samp-unsafe $N_SAMP_UNSAFE --n-samp-safe $N_SAMP_SAFE --n-samp-all $N_SAMP_ALL \
    --dual-scheme $DUAL_SCHEME \
    --robust --lambda-robust $LAMBDA_ROBUST 
