import os

import ray


@ray.remote
def get_visible_gpus(device_control_env_var: str):
    cvd = os.environ.get(device_control_env_var, "")
    if cvd:
        return cvd.split(",")
    # Fallback: when RAY_EXPERIMENTAL_NOSET prevents Ray from setting device visibility,
    # use Ray's internal GPU assignment instead.
    return [str(gpu_id) for gpu_id in ray.get_gpu_ids()]


@ray.remote
def get_node_rank():
    return int(os.environ.get("NODE_RANK", "0"))
