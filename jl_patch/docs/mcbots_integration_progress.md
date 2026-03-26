# McbotsEnvManager Implementation Progress

## Plan: /homes/junlong/.claude/plans/mutable-hatching-naur.md

## Tasks

- [x] 1. `mcbots_env_manager.py` — Main env manager class
  - [x] 1a. Constructor (__init__)
  - [x] 1b. run_rollout_loop
  - [x] 1c. HTTP server (start, stop, auto port discovery, port file)
  - [x] 1d. _handle_chat_completion
  - [x] 1e. _handle_window_complete
  - [x] 1f. _handle_episode_done
  - [x] 1g. _formulate_dataproto (with response_token_ranges)
  - [x] 1h. _extract_and_convert_images
  - [x] 1i. Message recording logic with assertions
- [x] 2. `examples/config/mcbots_envs.yaml` — Config template
- [x] 3. `jl_patch/qwen3_5_mcbots.yaml` — Example pipeline config
- [ ] 4. Unit tests (Phase 1)
- [ ] 5. Integration tests (Phase 2)

## Files Created

- `roll/pipeline/agentic/env_manager/mcbots_env_manager.py` — Main implementation (~450 lines)
- `examples/config/mcbots_envs.yaml` — Hydra config template
- `jl_patch/qwen3_5_mcbots.yaml` — Example pipeline config for Qwen3.5-4B + Minecraft

## Key Design Points

- Auto port discovery via `socket.bind(('', 0))`
- Port written to `/tmp/roll_mcbots_ports/{env_id}.port` for mcbots discovery
- Assertions verify message consistency (only after /window_complete resets)
- No indirect context reset detection — only explicit /window_complete signal
- response_token_ranges in non_tensor_batch for easy response span lookup

## Log

- 2026-03-19: Created all 3 files. Core implementation complete. Next: unit tests.
