#!/bin/bash
set +x

unset VIRTUAL_ENV

export ROLL_RAY_RUNTIME_ENV=0
export MCBOTS_INITIAL_USER_INPUT="explore freely!"

# todo, may be 
export MC_SERVER_HOST=${ARNOLD_MCSERVER_0_HOST}
mc_server_runtime_config_file="/mnt/hdfs/tiktok_aiic/user/junlongli/mcbots/shared/server-runtime-${ARNOLD_TRIAL_START_TIME}.json"

# we wait the above file created then proceed, max wait time is 15 mins
MAX_WAIT_TIME=900
start_time=$(date +%s)
elapsed_time=0
while [ ! -f "$mc_server_runtime_config_file" ]; do
    sleep 1
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    if [ $elapsed_time -gt $MAX_WAIT_TIME ]; then
        echo "Error: MC server runtime config file not found after $MAX_WAIT_TIME seconds"
        exit 1
    fi
done

echo "MC server runtime config file found after ${elapsed_time} seconds , proceed to get MC_SERVER_PORT"
export MC_SERVER_PORT=$(python3 -c "import json; print(json.load(open('${mc_server_runtime_config_file}'))['server']['port'])")                                                

echo "MC_SERVER_HOST: ${MC_SERVER_HOST}"
echo "MC_SERVER_PORT: ${MC_SERVER_PORT}"

python examples/start_agentic_pipeline.py --config_path ../jl_patch_tt --config_name qwen3_5_4B_agentic_mcbots_multinode
