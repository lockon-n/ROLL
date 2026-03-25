#!/bin/bash

# start_ray.sh
# Usage: bash start_ray.sh <train_script.sh> [args...]

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <train_script.sh> [args...]"
    exit 1
fi

TRAIN_SCRIPT=$1
shift

# Environment variables setup (defaults if not provided)
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-6379}
RAY_TIMEOUT=${RAY_TIMEOUT:-1000}
VLLM_USE_V1=${VLLM_USE_V1:-1}

if [[ "$NNODES" -le 1 ]]; then
    # Single node execution
    echo "Single node detected, running $TRAIN_SCRIPT directly..."
    bash "$TRAIN_SCRIPT" "$@"
else
    # Multi-node execution using Ray
    if [[ "$NODE_RANK" == "0" ]]; then
        # Head node
        echo "Starting Ray head on $MASTER_ADDR..."
        ray stop --force
        ray start --head --node-ip-address="$MASTER_ADDR" \
                  --num-gpus="$NGPUS_PER_NODE" \
                  --disable-usage-stats \
                  --port="$MASTER_PORT" \
                  --min-worker-port=0 \
                  --max-worker-port=0

        # Build runtime env
        RUNTIME_ENV_JSON="{
        \"working_dir\": \".\",
        \"excludes\": [
            \"output\",
            \"output/**\",
            \"wandb\",
            \"wandb/**\",
            \"data\",
            \"data/**\",
            \"*.pdf\",
            \"**/*.pdf\"
        ],
        \"env_vars\": {
            \"VLLM_USE_V1\": \"$VLLM_USE_V1\"
        }
        }"

        echo "Ray head started"
        echo "Waiting for $NNODES nodes to join Ray cluster..."
        
        elapsed=0
        while true; do
            if [ "$elapsed" -ge 1000 ]; then
                echo "Error: Timeout reached after $elapsed seconds."
                echo "Cluster state at timeout:"
                ray list nodes
                exit 1
            fi

            echo "Checking Ray cluster status (Elapsed: ${elapsed}s)..."
            ray_output=""
            cmd_success=false

            # Try 3 times to list nodes
            for attempt in 1 2 3; do
                if ray_output=$(timeout 10 ray list nodes 2>&1); then
                    cmd_success=true
                    break
                else
                    echo "Attempt $attempt to list nodes failed (exit code: $?)"
                    [ $attempt -lt 3 ] && sleep 3
                fi
            done

            if [ "$cmd_success" = false ]; then
                echo "Warning: All attempts to run 'ray list nodes' failed"
                echo "Last error output: $ray_output"
                echo "Retrying in 10 seconds..."
                sleep 10
                elapsed=$((elapsed + 10))
                continue
            fi

            # Count nodes in ALIVE state
            alive_node_count=$(echo "$ray_output" | grep "ALIVE" -c || echo "0")
            echo "Progress: $alive_node_count / $NNODES nodes are ALIVE"

            if [ "$alive_node_count" -ge "$NNODES" ]; then
                echo "Success: Ray cluster is ready with $alive_node_count nodes!"
                break
            fi

            echo "Waiting 5 seconds before next check..."
            sleep 5
            elapsed=$((elapsed + 5))
        done

        # Escape each argument for the remote shell to preserve quotes
        ESCAPED_ARGS=()
        for arg in "$@"; do
            ESCAPED_ARGS+=("$(printf %q "$arg")")
        done

        echo "Submitting training job: bash $TRAIN_SCRIPT ${ESCAPED_ARGS[*]}"
        ray job submit --address="[$MASTER_ADDR]:$MASTER_PORT" \
            --runtime-env-json="$RUNTIME_ENV_JSON" \
            -- bash "$TRAIN_SCRIPT" "${ESCAPED_ARGS[@]}"
    else
        # Worker node
        echo "Starting Ray worker node, connecting to $MASTER_ADDR:$MASTER_PORT..."
        ray stop --force

        while true; do
            if ray status --address="[$MASTER_ADDR]:$MASTER_PORT" &>/dev/null; then
                echo "Found Ray head node"
                break
            fi
            echo "Waiting 5s for head node initialization..."
            sleep 5s
        done

        ray start --address="[$MASTER_ADDR]:$MASTER_PORT" \
                  --num-gpus "$NGPUS_PER_NODE" \
                  --min-worker-port=0 \
                  --max-worker-port=0 \
                  --block
    fi
fi
