#!/usr/bin/env python3

import argparse
import signal
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Keep a GPU busy with repeated random matrix multiplications."
    )
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cuda:0.")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Matrix dtype.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=8192,
        help="Square matrix size. Larger values increase load and memory use.",
    )
    parser.add_argument(
        "--buffers",
        type=int,
        default=3,
        help="How many matrix triplets to keep resident on the device.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations before statistics.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=50,
        help="Print one status line every N iterations.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep between iterations in seconds.",
    )
    parser.add_argument(
        "--util",
        type=float,
        default=None,
        help="Best-effort target GPU utilization percentage in (0, 100].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed.",
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="Allow TF32 on Ampere+ when using float32.",
    )
    return parser.parse_args()


def resolve_dtype(torch_mod, dtype_name):
    return {
        "float16": torch_mod.float16,
        "bfloat16": torch_mod.bfloat16,
        "float32": torch_mod.float32,
    }[dtype_name]


def make_buffers(torch_mod, device, dtype, size, buffers):
    mats = []
    for _ in range(buffers):
        a = torch_mod.randn((size, size), device=device, dtype=dtype)
        b = torch_mod.randn((size, size), device=device, dtype=dtype)
        c = torch_mod.empty((size, size), device=device, dtype=dtype)
        mats.append((a, b, c))
    return mats


def sleep_for_target_util(busy_time_s, target_util):
    if target_util is None or target_util >= 1.0:
        return 0.0
    return busy_time_s * ((1.0 / target_util) - 1.0)


def main():
    args = parse_args()

    if args.sleep < 0:
        print("--sleep must be >= 0.", file=sys.stderr)
        return 2

    if args.util is not None and not (0.0 < args.util <= 100.0):
        print("--util must be in the range (0, 100].", file=sys.stderr)
        return 2

    if args.util is not None and args.sleep > 0:
        print("--util cannot be combined with --sleep.", file=sys.stderr)
        return 2

    try:
        import torch
    except ImportError:
        print("PyTorch is required: pip install torch", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("CUDA is not available. This script is intended for GPU use.", file=sys.stderr)
        return 2

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device)
    dtype = resolve_dtype(torch, args.dtype)
    target_util = None if args.util is None else (args.util / 100.0)

    try:
        props = torch.cuda.get_device_properties(device)
        name = props.name
        total_mem_gb = props.total_memory / (1024 ** 3)
    except Exception:
        name = str(device)
        total_mem_gb = 0.0

    print(
        f"Starting GPU burner on {device} ({name}), "
        f"dtype={args.dtype}, size={args.size}, buffers={args.buffers}"
    )
    if target_util is not None:
        print(f"Target GPU utilization: {args.util:.1f}% (best-effort pacing)")
    if total_mem_gb:
        print(f"GPU total memory: {total_mem_gb:.2f} GiB")

    stop = False

    def handle_stop(signum, _frame):
        nonlocal stop
        stop = True
        print(f"\nReceived signal {signum}, stopping...")

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    matrices = make_buffers(torch, device, dtype, args.size, args.buffers)

    for i in range(args.warmup):
        a, b, c = matrices[i % len(matrices)]
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize(device)

    iteration = 0
    started_at = time.perf_counter()
    last_report_at = started_at
    interval_busy_time = 0.0
    pace_with_util = target_util is not None and target_util < 1.0
    start_event = end_event = None

    if pace_with_util:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    while not stop:
        a, b, c = matrices[iteration % len(matrices)]
        if pace_with_util:
            start_event.record()
            torch.matmul(a, b, out=c)
            end_event.record()
            end_event.synchronize()
            busy_time = start_event.elapsed_time(end_event) / 1000.0
            interval_busy_time += busy_time
        else:
            torch.matmul(a, b, out=c)
        iteration += 1

        if pace_with_util:
            sleep_time = sleep_for_target_util(busy_time, target_util)
            if sleep_time > 0:
                time.sleep(sleep_time)
        elif args.sleep > 0:
            torch.cuda.synchronize(device)
            time.sleep(args.sleep)

        if iteration % args.report_every == 0:
            if not pace_with_util:
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            elapsed = now - started_at
            interval = now - last_report_at
            it_per_sec = args.report_every / interval if interval > 0 else 0.0
            tflops = (
                (2.0 * (args.size ** 3) * it_per_sec) / 1e12
                if args.size > 0
                else 0.0
            )
            allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
            status = (
                f"[{elapsed:9.1f}s] iter={iteration:8d} "
                f"rate={it_per_sec:7.2f}/s approx={tflops:7.2f} TFLOPS "
                f"mem={allocated_gb:6.2f}/{reserved_gb:6.2f} GiB"
            )
            if pace_with_util:
                actual_util = (100.0 * interval_busy_time / interval) if interval > 0 else 0.0
                status += f" util={actual_util:5.1f}% target={args.util:5.1f}%"
                interval_busy_time = 0.0
            print(status)
            last_report_at = now

    torch.cuda.synchronize(device)
    print(f"Stopped after {iteration} iterations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
