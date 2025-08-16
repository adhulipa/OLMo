#!/bin/bash

# Configuration
CONFIG_PATH="configs/tiny/OLMo-20M.yaml"
NUM_GPUS=1  # Change this based on your setup
MASTER_PORT=12355

# Check if torchrun is available
if command -v torchrun &> /dev/null; then
    echo "Using torchrun for distributed training..."
    torchrun --nproc_per_node=$NUM_GPUS scripts/train.py $CONFIG_PATH --save_overwrite
else
    echo "torchrun not found. Setting up manual environment variables..."
    
    # Set environment variables for single GPU training
    export WORLD_SIZE=$NUM_GPUS
    export RANK=0
    export MASTER_ADDR=localhost
    export MASTER_PORT=$MASTER_PORT
    export LOCAL_RANK=0
    
    echo "Environment variables set:"
    echo "WORLD_SIZE=$WORLD_SIZE"
    echo "RANK=$RANK"
    echo "MASTER_ADDR=$MASTER_ADDR"
    echo "MASTER_PORT=$MASTER_PORT"
    echo "LOCAL_RANK=$LOCAL_RANK"
    
    python scripts/train.py $CONFIG_PATH --save_overwrite
fi