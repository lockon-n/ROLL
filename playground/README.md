# Ray Runtime Env Playground

This folder contains a minimal probe for the Ray `runtime_env` failure you saw.

## Files

- `ray_runtime_env_probe.py`: calls `ray.init(..., runtime_env={"env_vars": ...})`, then runs one remote task to confirm the runtime env is usable.

## Examples

Use the current local cluster address:

```bash
python playground/ray_runtime_env_probe.py --address 127.0.0.1:6379
```

Use the local IPv4 with an explicit node address:

```bash
python playground/ray_runtime_env_probe.py \
  --address 172.18.0.5:6379 \
  --node-ip-address 172.18.0.5
```

Use the local IPv6 with an explicit node address:

```bash
python playground/ray_runtime_env_probe.py \
  --address '[fdbd:fdbd:fdbd:fdbd:ffff:ffff:0:5]:6379' \
  --node-ip-address 'fdbd:fdbd:fdbd:fdbd:ffff:ffff:0:5'
```

Use the Arnold head IPv6 as the cluster address:

```bash
python playground/ray_runtime_env_probe.py \
  --address '[2605:340:cda2:1235:2bc5:48bf:daf:cfd5]:6379'
```

## What to look for

Success means:

- `ray.init() succeeded`
- `remote task succeeded`
- the remote result contains your probe env var

Failure means the script prints the exception type, message, and traceback. If the original issue is reproduced, you should see a `RuntimeEnvSetupError` and the `403 Missing Destination-Service header` message.
