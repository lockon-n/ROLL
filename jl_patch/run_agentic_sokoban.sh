#!/bin/bash
set +x

unset VIRTUAL_ENV

# pick X free ports from 20000 to 65535
function get_free_ports() {
    local n=$1
    comm -23 \
        <(seq 20000 65535 | sort) \
        <(ss -Htan | awk '{print $4}' | rev | cut -d':' -f1 | rev | sort -u) \
    | shuf | head -n "$n"
}

read MASTER_PORT DASHBOARD_PORT < <(get_free_ports 2 | tr '\n' ' ')

export RAY_TMPDIR=/ssddata/junlong/ray_tmp
export TMPDIR=/ssddata/junlong/tmp
export CUDA_HOME=~/cuda-12.9
export LD_LIBRARY_PATH=~/cuda-12.9/lib64:/export/ssddata/junlong/projects/ROLL/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/export/ssddata/junlong/projects/ROLL/.venv/lib/python3.12/site-packages/nvidia/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export WANDB_API_KEY=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
export ROLL_RAY_RUNTIME_ENV=1
export MASTER_PORT=${MASTER_PORT}
export DASHBOARD_PORT=${DASHBOARD_PORT}

echo "MASTER_PORT: ${MASTER_PORT}"
echo "DASHBOARD_PORT: ${DASHBOARD_PORT}"

ray stop 2>/dev/null

uv run examples/start_agentic_pipeline.py --config_path ../jl_patch --config_name qwen3_5_0_8B_agentic_sokoban
