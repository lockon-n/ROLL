#!/usr/bin/env python3
import argparse
import os
import sys
import traceback

import ray


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal probe for Ray runtime_env setup failures with different address settings."
    )
    parser.add_argument(
        "--address",
        default=None,
        help="Ray cluster address, for example 127.0.0.1:6379 or [ipv6]:6379. If omitted, Ray starts/connects locally.",
    )
    parser.add_argument(
        "--node-ip-address",
        dest="node_ip_address",
        default=None,
        help="Optional _node_ip_address passed to ray.init().",
    )
    parser.add_argument(
        "--env-key",
        default="RAY_RUNTIME_ENV_PROBE",
        help="Key injected through runtime_env.env_vars.",
    )
    parser.add_argument(
        "--env-value",
        default="1",
        help="Value injected through runtime_env.env_vars.",
    )
    return parser.parse_args()


@ray.remote
def read_env(key: str):
    return {
        "pid": os.getpid(),
        "hostname": os.uname().nodename,
        "value": os.getenv(key),
    }


def main():
    args = parse_args()

    init_kwargs = {
        "address": args.address,
        "ignore_reinit_error": True,
        "log_to_driver": True,
        "runtime_env": {
            "env_vars": {
                args.env_key: args.env_value,
            }
        },
    }
    if args.node_ip_address:
        init_kwargs["_node_ip_address"] = args.node_ip_address

    print("=== Probe Configuration ===")
    print(f"address={args.address!r}")
    print(f"node_ip_address={args.node_ip_address!r}")
    print(f"runtime_env.env_vars={{{args.env_key!r}: {args.env_value!r}}}")
    print()

    try:
        ray.init(**init_kwargs)
        print("ray.init() succeeded")
        print(f"ray.__version__={ray.__version__}")
        print(f"ray.nodes()={ray.nodes()}")

        result = ray.get(read_env.remote(args.env_key))
        print("remote task succeeded")
        print(f"remote result={result}")
        return 0
    except Exception as exc:
        print("ray probe failed")
        print(f"exception_type={type(exc).__name__}")
        print(f"exception={exc}")
        print()
        traceback.print_exc()
        return 1
    finally:
        try:
            ray.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
