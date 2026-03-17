import os
from typing import Any, Optional


def is_single_node_mode() -> bool:
    try:
        return int(os.getenv("ARNOLD_WORKER_NUM", "0")) == 1
    except ValueError:
        return False


def build_ray_init_kwargs(address: Optional[str] = None, **kwargs: Any) -> dict[str, Any]:
    init_kwargs = dict(kwargs)
    init_kwargs["address"] = address

    # In single-node mode, letting Ray infer a Pod IP can route the local
    # runtime_env agent HTTP traffic through the compliance gateway.
    if address is None and is_single_node_mode():
        init_kwargs["_node_ip_address"] = "127.0.0.1"

    return init_kwargs
