"""Minimal script to trace num_gpus_per_node assignment chain."""
import sys
sys.path.insert(0, ".")

# Step 1: dataclass default
from roll.configs.base_config import PPOConfig
import dataclasses
default_val = dataclasses.fields(PPOConfig)[
    [f.name for f in dataclasses.fields(PPOConfig)].index("num_gpus_per_node")
].default
print(f"[Step 1] dataclass default: num_gpus_per_node = {default_val}")

# Step 2: Hydra config resolution
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize_config_dir
import os

config_dir = os.path.abspath("jl_patch_tt")
with initialize_config_dir(config_dir=config_dir, version_base=None):
    cfg = compose(config_name="qwen3_5_0_8B_agentic_sokoban_multinode")
print(f"[Step 2] Hydra yaml value: num_gpus_per_node = {cfg.num_gpus_per_node}")

# Step 3: device_count() on this node
from roll.platforms import current_platform
detected_gpus = current_platform.device_count()
print(f"[Step 3] current_platform.device_count() = {detected_gpus}")

# Step 4: override logic
yaml_val = cfg.num_gpus_per_node
if detected_gpus > 0:
    final_val = detected_gpus
    print(f"[Step 4] detected_gpus > 0, OVERRIDE: num_gpus_per_node = {yaml_val} -> {final_val}")
else:
    final_val = yaml_val
    print(f"[Step 4] detected_gpus == 0, NO override: num_gpus_per_node = {final_val}")

# Step 5: num_nodes calculation
device_mappings = {
    "actor_train": list(range(0, 4)),
    "actor_infer": list(range(0, 4)),
    "reference": list(range(0, 4)),
}
total_devices = []
for name, mapping in device_mappings.items():
    total_devices.extend(mapping)
    print(f"  [{name}] device_mapping = {mapping}")

max_gpu_num = max(total_devices) + 1
print(f"[Step 5] max_gpu_num = max({set(total_devices)}) + 1 = {max_gpu_num}")

if max_gpu_num <= final_val:
    num_nodes = 1
else:
    num_nodes = (max_gpu_num + final_val - 1) // final_val
print(f"[Step 5] num_nodes = ({max_gpu_num} + {final_val} - 1) // {final_val} = {num_nodes}")

# Step 6: what ResourceManager would see
print(f"\n[Step 6] ResourceManager(num_nodes={num_nodes}, num_gpus_per_node={final_val})")
print(f"  -> needs {num_nodes} nodes with >= {final_val} GPU(s) each")
print(f"  -> you have 2 workers with 2 GPUs each => ray_num_nodes = 2")
print(f"  -> assert {num_nodes} <= 2 => {'PASS' if num_nodes <= 2 else 'FAIL'}")
