#!/bin/bash
set +x

unset VIRTUAL_ENV

# pip install dacite
# pip install imageio
# pip install gem-llm==0.0.4
# pip install transformers==5.2.0
# pip install flash-linear-attention==0.4.2
# pip uninstall byted-wandb
# pip uninstall bytedray
# pip install ray
# pip install "antlr4-python3-runtime==4.9.3" "latex2sympy2_extended==1.10.1"
# pip install -e mcore_adapter

# Stop any residual Ray cluster to avoid stale GPU mappings
ray stop 2>/dev/null

python examples/start_agentic_pipeline.py --config_path ../jl_patch_tt --config_name qwen3_5_0_8B_agentic_sokoban
