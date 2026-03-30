"""Minimal script to trace num_gpus_per_node assignment chain via roll imports."""
import os
import sys
import logging.handlers  # must import before roll to avoid AttributeError

sys.path.insert(0, ".")

print(f"hostname = {os.uname().nodename}")
print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

# Step 1: dataclass default
import dataclasses
from roll.configs.base_config import PPOConfig

default_val = dataclasses.fields(PPOConfig)[
    [f.name for f in dataclasses.fields(PPOConfig)].index("num_gpus_per_node")
].default
print(f"\n[Step 1] PPOConfig dataclass default: num_gpus_per_node = {default_val}")

# Step 2: current_platform.device_count()
from roll.platforms import current_platform

detected_gpus = current_platform.device_count()
print(f"[Step 2] current_platform.device_count() = {detected_gpus}")

import torch
torch_gpus = torch.cuda.device_count()
print(f"  torch.cuda.device_count() = {torch_gpus}")
print(f"  platform device_type = {current_platform.device_type}")

# Step 3: simulate override logic (base_config.py:327-329)
yaml_val = 2
if detected_gpus > 0:
    final_val = detected_gpus
    print(f"[Step 3] detected_gpus > 0 => OVERRIDE: num_gpus_per_node = {yaml_val} -> {final_val}")
else:
    final_val = yaml_val
    print(f"[Step 3] detected_gpus == 0 => NO override: num_gpus_per_node stays {final_val}")

# Step 4: num_nodes calculation (base_config.py:379-384)
device_mappings = {
    "actor_train": list(range(0, 4)),
    "actor_infer": list(range(0, 4)),
    "reference": list(range(0, 4)),
}
total_devices = []
for name, mapping in device_mappings.items():
    total_devices.extend(mapping)

max_gpu_num = max(total_devices) + 1
if max_gpu_num <= final_val:
    num_nodes = 1
else:
    num_nodes = (max_gpu_num + final_val - 1) // final_val
print(f"[Step 4] max_gpu_num={max_gpu_num}, num_gpus_per_node={final_val} => num_nodes={num_nodes}")

# Step 5: ResourceManager assertion check
print(f"\n[Step 5] ResourceManager(num_nodes={num_nodes}, num_gpus_per_node={final_val})")
print(f"  assert {num_nodes} <= 2 (ray_num_nodes) => {'PASS' if num_nodes <= 2 else 'FAIL'}")
