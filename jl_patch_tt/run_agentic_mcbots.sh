#!/bin/bash
set +x

unset VIRTUAL_ENV

# Stop any residual Ray cluster to avoid stale GPU mappings
ray stop 2>/dev/null

export ROLL_RAY_RUNTIME_ENV=0

python examples/start_agentic_pipeline.py --config_path ../jl_patch_tt --config_name qwen3_5_0_8B_agentic_mcbots
