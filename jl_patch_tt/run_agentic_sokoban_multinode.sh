#!/bin/bash
set +x

unset VIRTUAL_ENV

export ROLL_RAY_RUNTIME_ENV=1
export ROLL_PLATFORM=cuda

python examples/start_agentic_pipeline.py --config_path ../jl_patch_tt --config_name qwen3_5_0_8B_agentic_sokoban_multinode
