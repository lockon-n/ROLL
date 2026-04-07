#!/bin/bash
set +x

unset VIRTUAL_ENV

# TODO, implement a function to get N free ports
function get_free_ports() {
    local n=$1
    local ports=()
    for i in $(seq 1 $n); do
        ports+=($(comm -23 <(seq 1 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1))
    done
    echo ${ports[@]}
}

free_ports=$(get_free_ports 2)
MASTER_PORT=${free_ports[0]}
DASHBOARD_PORT=${free_ports[1]}

export ROLL_RAY_RUNTIME_ENV=1
export MASTER_PORT=${MASTER_PORT}
export DASHBOARD_PORT=${DASHBOARD_PORT}

echo "MASTER_PORT: ${MASTER_PORT}"
echo "DASHBOARD_PORT: ${DASHBOARD_PORT}"

ray stop 2>/dev/null

WANDB_API_KEY=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 \
TMPDIR=/export/ssddata/junlong/tmp \
RAY_TMPDIR=/export/ssddata/junlong/tmp/ray \
CUDA_HOME=~/cuda-12.9 \
LD_LIBRARY_PATH=~/cuda-12.9/lib64:/export/ssddata/junlong/projects/ROLL/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/export/ssddata/junlong/projects/ROLL/.venv/lib/python3.12/site-packages/nvidia/nccl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} \
uv run python examples/start_agentic_pipeline.py --config_path ../jl_patch --config_name qwen3_5_0_8B_agentic_mcbots
