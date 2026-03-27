import os
import subprocess
import sys
import time

import ray

from roll.distributed.ray_utils import build_ray_init_kwargs
from roll.distributed.scheduler.driver_utils import (
    get_driver_rank,
    get_driver_master_addr,
    get_driver_node_name,
    get_driver_master_port,
    get_driver_world_size,
    get_driver_dashboard_port,
    get_ray_status,
    is_ray_cluster_running,
    wait_for_nodes,
)
from roll.distributed.scheduler.log_monitor import LogMonitorListener
from roll.utils.constants import RAY_NAMESPACE
from roll.utils.logging import get_logger
from roll.platforms import current_platform

logger = get_logger()


def start_ray_cluster():
    rank = get_driver_rank()
    world_size = get_driver_world_size()
    master_addr = get_driver_master_addr()
    master_port = get_driver_master_port()
    node_name = get_driver_node_name()
    dashboard_port = get_driver_dashboard_port()

    if is_ray_cluster_running():
        logger.info("Ray cluster already initialized")
        return False

    if rank == 0:
        cmd = f"ray start --head --port={master_port} --node-name={node_name} --dashboard-port={dashboard_port}"
    else:
        # fix: 处理大规模下可能会出现的head/worker node创建顺序不一致问题
        time.sleep(5)
        cmd = f"ray start --address={master_addr}:{master_port} --node-name={node_name} --dashboard-port={dashboard_port}"

    logger.info(f"Starting ray cluster: {cmd}")
    ret = subprocess.run(cmd, shell=True, capture_output=True)
    if ret.returncode != 0:
        logger.error(f"Failed to start ray cluster: {cmd}")
        logger.error(f"ret.stdout: {ret.stdout}")
        logger.error(f"ret.stderr: {ret.stderr}")
        sys.exit(1)
    return True


def init():
    _t0 = time.time()
    rank = get_driver_rank()
    world_size = get_driver_world_size()
    master_addr = get_driver_master_addr()
    master_port = get_driver_master_port()

    platform_env = current_platform.get_custom_env_vars()
    os.environ.update(platform_env)

    use_runtime_env = os.environ.get("ROLL_RAY_RUNTIME_ENV", "0") == "1"
    if not use_runtime_env:
        # Prevent Ray from overriding device visibility env vars for actors.
        # Must be set before ray start / ray.init so raylet and all actors inherit it.
        os.environ.setdefault(current_platform.ray_experimental_noset, "1")

    _t1 = time.time()
    manual_start = start_ray_cluster()
    logger.info(f"[DIAG] start_ray_cluster took {time.time() - _t1:.1f}s")

    ray_init_kwargs = build_ray_init_kwargs(
        address=f"{master_addr}:{master_port}" if manual_start else None,
        namespace=RAY_NAMESPACE,
        ignore_reinit_error=True,
        log_to_driver=not manual_start,
    )
    if use_runtime_env:
        ray_init_kwargs["runtime_env"] = {"env_vars": platform_env}

    if not ray.is_initialized():
        _t1 = time.time()
        ray.init(**ray_init_kwargs)
        logger.info(f"[DIAG] ray.init took {time.time() - _t1:.1f}s")

    if manual_start:
        _t1 = time.time()
        wait_for_nodes(expected=world_size)
        logger.info(f"[DIAG] wait_for_nodes took {time.time() - _t1:.1f}s")
        listener = LogMonitorListener()
        listener.start()

    logger.info(f"Current ray cluster resources: {ray.available_resources()}")
    logger.info(f"[DIAG] === init() TOTAL took {time.time() - _t0:.1f}s ===")

    if manual_start and rank > 0:
        sys.exit(0)
