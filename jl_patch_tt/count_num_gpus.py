"""Minimal script to trace num_gpus_per_node assignment chain. No roll imports."""
import os
import torch

# Step 1: dataclass default
print(f"[Step 1] PPOConfig dataclass default: num_gpus_per_node = 1")

# Step 2: yaml config value (hardcoded from your yaml)
yaml_val = 2
print(f"[Step 2] Your yaml sets: num_gpus_per_node = {yaml_val}")

# Step 3: what device_count() sees on this node
detected_gpus = torch.cuda.device_count()
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
print(f"[Step 3] torch.cuda.device_count() = {detected_gpus}")
print(f"  CUDA_VISIBLE_DEVICES = {cuda_visible}")
print(f"  hostname = {os.uname().nodename}")

# Step 4: override logic (base_config.py:327-329)
if detected_gpus > 0:
    final_val = detected_gpus
    print(f"[Step 4] detected_gpus > 0 => OVERRIDE: num_gpus_per_node = {yaml_val} -> {final_val}")
else:
    final_val = yaml_val
    print(f"[Step 4] detected_gpus == 0 => NO override: num_gpus_per_node stays {final_val}")

# Step 5: num_nodes calculation (base_config.py:379-384)
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
print(f"[Step 5] max_gpu_num = {max_gpu_num}, num_gpus_per_node = {final_val}")

if max_gpu_num <= final_val:
    num_nodes = 1
else:
    num_nodes = (max_gpu_num + final_val - 1) // final_val
print(f"[Step 5] num_nodes = ({max_gpu_num} + {final_val} - 1) // {final_val} = {num_nodes}")

# Step 6: ResourceManager assertion check
print(f"\n[Step 6] ResourceManager would get: num_nodes={num_nodes}, num_gpus_per_node={final_val}")
print(f"  Needs {num_nodes} nodes with >= {final_val} GPU(s) each")
print(f"  You have 2 workers with 2 GPUs => ray_num_nodes = 2")
print(f"  assert {num_nodes} <= 2 => {'PASS' if num_nodes <= 2 else 'FAIL'}")
