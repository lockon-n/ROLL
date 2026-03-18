#!/bin/bash
set +x

unset VIRTUAL_ENV

# pip install dacite
# pip install imageio

# Stop any residual Ray cluster to avoid stale GPU mappings
/export/ssddata/junlong/projects/ROLL/.venv/bin/ray stop 2>/dev/null

WANDB_API_KEY=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 \
TMPDIR=/export/ssddata/junlong/tmp \
RAY_TMPDIR=/export/ssddata/junlong/tmp/ray \
CUDA_HOME=~/cuda-12.9 \
LD_LIBRARY_PATH=~/cuda-12.9/lib64:/export/ssddata/junlong/projects/ROLL/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/export/ssddata/junlong/projects/ROLL/.venv/lib/python3.12/site-packages/nvidia/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
uv run python examples/start_agentic_pipeline.py --config_path ../jl_patch --config_name qwen3_5_0_8B_agentic_sokoban
