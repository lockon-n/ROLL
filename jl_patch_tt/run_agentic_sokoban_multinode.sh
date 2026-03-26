#!/bin/bash
set +x

unset VIRTUAL_ENV
export ROLL_PLATFORM=cuda
# export NNODES=$ARNOLD_WORKER_NUM
# export NODE_RANK=$ARNOLD_ID
export RANK=$ARNOLD_ID
export WORLD_SIZE=$ARNOLD_WORKER_NUM
export WORKER_NUM=$ARNOLD_WORKER_NUM
export MASTER_ADDR=$ARNOLD_WORKER_0_HOST
export MASTER_PORT=$(echo $ARNOLD_WORKER_0_PORT | cut -d',' -f1)

python examples/start_agentic_pipeline.py --config_path ../jl_patch_tt --config_name qwen3_5_0_8B_agentic_sokoban_multinode
