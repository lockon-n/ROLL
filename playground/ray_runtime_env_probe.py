#!/usr/bin/env python3
import argparse
import os
import sys
import traceback
from typing import Dict, List

import ray


def parse_env_item(item: str) -> tuple[str, str]:
    if "=" not in item:
        raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got: {item!r}")
    key, value = item.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError(f"Empty env key is not allowed: {item!r}")
    return key, value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal probe for Ray runtime_env setup failures with different address settings."
    )
    parser.add_argument(
        "--mode",
        choices=["runtime_env", "process_env"],
        default="runtime_env",
        help="How to inject env vars. runtime_env uses Ray runtime_env; process_env updates os.environ before ray.init().",
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
    parser.add_argument(
        "--env",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Additional env var to propagate. Can be repeated.",
    )
    parser.add_argument(
        "--include-platform-env",
        action="store_true",
        help="Include current_platform.get_custom_env_vars() in the propagated env set.",
    )
    return parser.parse_args()


@ray.remote
def read_env(keys: List[str]):
    return {
        "pid": os.getpid(),
        "hostname": os.uname().nodename,
        "values": {key: os.getenv(key) for key in keys},
    }


@ray.remote
class EnvActor:
    def __init__(self, keys: List[str]):
        self.keys = keys

    def read(self):
        return {
            "pid": os.getpid(),
            "hostname": os.uname().nodename,
            "values": {key: os.getenv(key) for key in self.keys},
        }


def build_env_vars(args) -> Dict[str, str]:
    env_vars: Dict[str, str] = {}

    if args.include_platform_env:
        from roll.platforms import current_platform

        env_vars.update(current_platform.get_custom_env_vars())

    if args.env:
        for item in args.env:
            key, value = parse_env_item(item)
            env_vars[key] = value
    else:
        env_vars[args.env_key] = args.env_value

    return env_vars


def snapshot_env(keys: List[str]) -> Dict[str, str | None]:
    return {key: os.getenv(key) for key in keys}


def compare_envs(expected: Dict[str, str], observed: Dict[str, str | None]) -> Dict[str, Dict[str, str | None]]:
    mismatches: Dict[str, Dict[str, str | None]] = {}
    for key, expected_value in expected.items():
        observed_value = observed.get(key)
        if observed_value != expected_value:
            mismatches[key] = {
                "expected": expected_value,
                "observed": observed_value,
            }
    return mismatches


def main():
    args = parse_args()
    env_vars = build_env_vars(args)
    env_keys = list(env_vars.keys())

    init_kwargs = {
        "address": args.address,
        "ignore_reinit_error": True,
        "log_to_driver": True,
    }
    if args.mode == "runtime_env":
        init_kwargs["runtime_env"] = {
            "env_vars": env_vars,
        }
    else:
        os.environ.update(env_vars)

    if args.node_ip_address:
        init_kwargs["_node_ip_address"] = args.node_ip_address

    driver_env = snapshot_env(env_keys)

    print("=== Probe Configuration ===")
    print(f"mode={args.mode!r}")
    print(f"address={args.address!r}")
    print(f"node_ip_address={args.node_ip_address!r}")
    print(f"env_vars={env_vars}")
    print(f"driver_env={driver_env}")
    print()

    try:
        ray.init(**init_kwargs)
        print("ray.init() succeeded")
        print(f"ray.__version__={ray.__version__}")
        print(f"ray.nodes()={ray.nodes()}")

        task_result = ray.get(read_env.remote(env_keys))
        actor = EnvActor.remote(env_keys)
        actor_result = ray.get(actor.read.remote())

        print("remote task succeeded")
        print(f"remote task result={task_result}")
        print("remote actor succeeded")
        print(f"remote actor result={actor_result}")

        task_mismatches = compare_envs(env_vars, task_result["values"])
        actor_mismatches = compare_envs(env_vars, actor_result["values"])
        driver_mismatches = compare_envs(env_vars, driver_env)

        if args.mode == "process_env" and driver_mismatches:
            print(f"driver mismatches={driver_mismatches}")
            return 2
        if task_mismatches:
            print(f"task mismatches={task_mismatches}")
            return 2
        if actor_mismatches:
            print(f"actor mismatches={actor_mismatches}")
            return 2

        print("all expected env vars matched")
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
