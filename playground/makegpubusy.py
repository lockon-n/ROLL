#!/usr/bin/env python3

import argparse
import signal
import subprocess
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
        "--adaptive",
        action="store_true",
        help="Adaptive mode: monitor real GPU utilization and fill the gap. "
        "Must be combined with --util, which becomes the target *total* utilization. "
        "When GPU is already busy from other work, this script backs off automatically.",
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="Allow TF32 on Ampere+ when using float32.",
    )
    return parser.parse_args()


def _parse_device_index(device_str: str) -> int:
    """Extract GPU index from a device string like 'cuda', 'cuda:0', 'cuda:3'."""
    if ":" in device_str:
        return int(device_str.split(":")[1])
    return 0


def _init_nvml_handle(device_index: int):
    """Try to initialize pynvml and return a device handle, or None on failure."""
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(device_index), pynvml
    except Exception:
        return None, None


def _query_util_nvml(handle, pynvml_mod) -> float:
    """Query GPU utilization via pynvml. Returns 0-100."""
    rates = pynvml_mod.nvmlDeviceGetUtilizationRates(handle)
    return float(rates.gpu)


def _query_util_smi(device_index: int) -> float:
    """Fallback: query GPU utilization via nvidia-smi subprocess. Returns 0-100."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits", f"-i={device_index}"],
            timeout=5, text=True,
        )
        return float(out.strip())
    except Exception:
        return 0.0


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

    if args.adaptive and args.util is None:
        print("--adaptive requires --util to set target total utilization.", file=sys.stderr)
        return 2

    if args.adaptive and args.sleep > 0:
        print("--adaptive cannot be combined with --sleep.", file=sys.stderr)
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
    if args.adaptive:
        print(f"Adaptive mode: target total GPU utilization = {args.util:.1f}%")
    elif target_util is not None:
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

    warmup_event_s = torch.cuda.Event(enable_timing=True)
    warmup_event_e = torch.cuda.Event(enable_timing=True)
    warmup_times = []
    for i in range(args.warmup):
        a, b, c = matrices[i % len(matrices)]
        warmup_event_s.record()
        torch.matmul(a, b, out=c)
        warmup_event_e.record()
        warmup_event_e.synchronize()
        warmup_times.append(warmup_event_s.elapsed_time(warmup_event_e) / 1000.0)
    matmul_time = sum(warmup_times) / len(warmup_times) if warmup_times else 0.005
    print(f"Measured matmul time: {matmul_time * 1000:.2f} ms")

    iteration = 0
    started_at = time.perf_counter()
    last_report_at = started_at
    interval_busy_time = 0.0
    pace_with_util = target_util is not None and target_util < 1.0 and not args.adaptive
    start_event = end_event = None

    if pace_with_util:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    # --- Adaptive mode state ---
    adaptive = args.adaptive
    if adaptive:
        device_index = _parse_device_index(args.device)
        nvml_handle, pynvml_mod = _init_nvml_handle(device_index)
        if nvml_handle is not None:
            query_util = lambda: _query_util_nvml(nvml_handle, pynvml_mod)
            print(f"Adaptive: using pynvml for utilization queries (device {device_index})")
        else:
            query_util = lambda: _query_util_smi(device_index)
            print(f"Adaptive: using nvidia-smi fallback for utilization queries (device {device_index})")
        duty_cycle = 1.0  # fraction of time doing matmuls (0.05 ~ 1.0)
        adaptive_sleep = 0.0
        last_sample_time = time.perf_counter()
        sample_interval = 1.0  # match nvidia's ~1s reporting window
        smoothed_util = float(query_util())
        last_gpu_util = smoothed_util
        ema_alpha = 0.3
        dead_zone = 3.0  # ±3% dead zone
        duty_gain = 0.008  # duty cycle adjustment per 1% gap

    while not stop:
        # --- Adaptive: sample GPU util and adjust duty cycle ---
        if adaptive:
            now_t = time.perf_counter()
            if now_t - last_sample_time >= sample_interval:
                raw_util = query_util()
                smoothed_util = ema_alpha * raw_util + (1.0 - ema_alpha) * smoothed_util
                last_gpu_util = smoothed_util
                last_sample_time = now_t
                gap = smoothed_util - args.util  # positive = too busy
                if abs(gap) > dead_zone:
                    # Decrease duty when over target, increase when under
                    duty_cycle = max(min(duty_cycle - gap * duty_gain * 0.01, 1.0), 0.02)
                # Convert duty cycle to sleep time
                if duty_cycle < 1.0:
                    adaptive_sleep = matmul_time * (1.0 / duty_cycle - 1.0)
                else:
                    adaptive_sleep = 0.0

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

        if adaptive:
            if adaptive_sleep > 0:
                torch.cuda.synchronize(device)
                time.sleep(adaptive_sleep)
        elif pace_with_util:
            sleep_time = sleep_for_target_util(busy_time, target_util)
            if sleep_time > 0:
                time.sleep(sleep_time)
        elif args.sleep > 0:
            torch.cuda.synchronize(device)
            time.sleep(args.sleep)

        if iteration % args.report_every == 0:
            if not pace_with_util and not adaptive:
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
            if adaptive:
                status += (
                    f" gpu_util={last_gpu_util:5.1f}% target={args.util:5.1f}%"
                    f" duty={duty_cycle:.1%}"
                )
            elif pace_with_util:
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
