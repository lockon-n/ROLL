#!/bin/sh
set -eu

warn_count=0
fail_count=0
log_file=""

usage() {
    cat <<'EOF'
Usage:
  sh jl_patch_tt/verify_cache.sh [--log /path/to/vllm.log]

What it checks:
  1. Current user and HOME
  2. Restored Triton/vLLM cache directories
  3. Presence of compiled vLLM artifacts
  4. Python package versions and GPU model
  5. Optional log scan for cache hit/miss lines
EOF
}

say() {
    printf '%s\n' "$*"
}

section() {
    printf '\n== %s ==\n' "$1"
}

warn() {
    warn_count=$((warn_count + 1))
    printf '[warn] %s\n' "$1"
}

fail() {
    fail_count=$((fail_count + 1))
    printf '[fail] %s\n' "$1"
}

ok() {
    printf '[ok] %s\n' "$1"
}

count_files() {
    target_dir="$1"
    pattern="$2"
    if [ -d "$target_dir" ]; then
        find "$target_dir" -type f -name "$pattern" 2>/dev/null | wc -l | awk '{print $1}'
    else
        echo 0
    fi
}

list_some() {
    target_dir="$1"
    pattern="$2"
    limit="$3"
    if [ -d "$target_dir" ]; then
        find "$target_dir" -type f -name "$pattern" 2>/dev/null | sed -n "1,${limit}p"
    fi
}

while [ $# -gt 0 ]; do
    case "$1" in
        --log)
            shift
            if [ $# -eq 0 ]; then
                echo "--log requires a file path" >&2
                exit 2
            fi
            log_file="$1"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

CURRENT_HOME="${HOME:-}"
TARGET_HOME="/home/tiger"
CURRENT_VLLM_CACHE=""
CURRENT_TRITON_CACHE=""
TARGET_VLLM_CACHE="${TARGET_HOME}/.cache/vllm"
TARGET_TRITON_CACHE="${TARGET_HOME}/.triton/cache"

if [ -n "$CURRENT_HOME" ]; then
    CURRENT_VLLM_CACHE="${CURRENT_HOME}/.cache/vllm"
    CURRENT_TRITON_CACHE="${CURRENT_HOME}/.triton/cache"
fi

section "Identity"
if command -v whoami >/dev/null 2>&1; then
    say "user: $(whoami)"
fi
say "HOME: ${CURRENT_HOME:-<unset>}"
say "pwd: $(pwd)"

section "Cache Directories"
if [ -n "$CURRENT_HOME" ] && [ -d "$CURRENT_VLLM_CACHE" ]; then
    ok "current HOME vLLM cache exists: $CURRENT_VLLM_CACHE"
else
    warn "current HOME vLLM cache missing: ${CURRENT_VLLM_CACHE:-<unset>}"
fi

if [ -n "$CURRENT_HOME" ] && [ -d "$CURRENT_TRITON_CACHE" ]; then
    ok "current HOME Triton cache exists: $CURRENT_TRITON_CACHE"
else
    warn "current HOME Triton cache missing: ${CURRENT_TRITON_CACHE:-<unset>}"
fi

if [ -d "$TARGET_VLLM_CACHE" ]; then
    ok "restored target vLLM cache exists: $TARGET_VLLM_CACHE"
else
    fail "restored target vLLM cache missing: $TARGET_VLLM_CACHE"
fi

if [ -d "$TARGET_TRITON_CACHE" ]; then
    ok "restored target Triton cache exists: $TARGET_TRITON_CACHE"
else
    fail "restored target Triton cache missing: $TARGET_TRITON_CACHE"
fi

section "Compiled Artifacts"
target_vllm_compile_count="$(count_files "$TARGET_VLLM_CACHE" "vllm_compile_cache.py")"
target_key_count="$(count_files "$TARGET_VLLM_CACHE" "cache_key_factors.json")"
target_graph_count="$(count_files "$TARGET_VLLM_CACHE" "computation_graph.py")"
target_artifact_count="$(count_files "$TARGET_VLLM_CACHE" "artifact_compile_range_*")"

say "target vllm_compile_cache.py count: $target_vllm_compile_count"
say "target cache_key_factors.json count: $target_key_count"
say "target computation_graph.py count: $target_graph_count"
say "target artifact_compile_range_* count: $target_artifact_count"

if [ "$target_vllm_compile_count" -eq 0 ]; then
    fail "no vLLM compiled graph cache found under $TARGET_VLLM_CACHE"
else
    ok "found vLLM compiled graph cache under $TARGET_VLLM_CACHE"
fi

if [ "$target_key_count" -eq 0 ]; then
    warn "no cache_key_factors.json found under $TARGET_VLLM_CACHE"
else
    ok "found cache_key_factors.json under $TARGET_VLLM_CACHE"
fi

say "sample cache files:"
list_some "$TARGET_VLLM_CACHE" "vllm_compile_cache.py" 5
list_some "$TARGET_VLLM_CACHE" "cache_key_factors.json" 5

if [ -n "$CURRENT_HOME" ] && [ "$CURRENT_HOME" != "$TARGET_HOME" ]; then
    current_vllm_compile_count="$(count_files "$CURRENT_VLLM_CACHE" "vllm_compile_cache.py")"
    current_key_count="$(count_files "$CURRENT_VLLM_CACHE" "cache_key_factors.json")"

    say "current HOME vllm_compile_cache.py count: $current_vllm_compile_count"
    say "current HOME cache_key_factors.json count: $current_key_count"

    if [ "$current_vllm_compile_count" -eq 0 ] && [ "$target_vllm_compile_count" -gt 0 ]; then
        warn "cache exists under $TARGET_HOME but not under current HOME; this usually means HOME mismatch"
    fi
fi

section "Versions"
py_bin=""
if command -v python >/dev/null 2>&1; then
    py_bin="python"
elif command -v python3 >/dev/null 2>&1; then
    py_bin="python3"
fi

if [ -n "$py_bin" ]; then
    "$py_bin" - <<'PY'
import os

def show_import(name):
    try:
        module = __import__(name)
        version = getattr(module, "__version__", "<unknown>")
        print(f"{name}: {version}")
        return module
    except Exception as exc:
        print(f"{name}: import failed: {exc}")
        return None

torch = show_import("torch")
show_import("vllm")
show_import("triton")

if torch is not None:
    print(f"torch.cuda: {getattr(torch.version, 'cuda', None)}")

for key in (
    "HOME",
    "VLLM_CACHE_ROOT",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "VLLM_DISABLE_COMPILE_CACHE",
):
    print(f"{key}: {os.getenv(key)}")
PY
else
    warn "python/python3 not found; skipped version checks"
fi

section "GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
else
    warn "nvidia-smi not found; skipped GPU checks"
fi

if [ -n "$log_file" ]; then
    section "Log Scan"
    if [ ! -f "$log_file" ]; then
        fail "log file not found: $log_file"
    else
        say "log file: $log_file"
        if command -v rg >/dev/null 2>&1; then
            rg -n 'Using cache directory|Directly load the compiled graph|Compiling a graph' "$log_file" || true
        else
            grep -nE 'Using cache directory|Directly load the compiled graph|Compiling a graph' "$log_file" || true
        fi

        if grep -q 'Directly load the compiled graph' "$log_file"; then
            ok "log shows a direct cache load"
        elif grep -q 'Compiling a graph' "$log_file"; then
            warn "log shows graph compilation; cache was not fully reused"
        else
            warn "log does not contain an obvious cache hit/miss marker"
        fi
    fi
fi

section "Summary"
if [ "$fail_count" -gt 0 ]; then
    say "result: FAIL ($fail_count fail, $warn_count warn)"
    say "next step: if HOME is correct, compare new-machine cache key and package/runtime versions"
    exit 1
fi

if [ "$warn_count" -gt 0 ]; then
    say "result: WARN (0 fail, $warn_count warn)"
    say "next step: check HOME and optional log output, then compare version/GPU/hash inputs"
    exit 0
fi

say "result: OK (0 fail, 0 warn)"
say "cache restore looks structurally correct; if runtime still recompiles, the key likely changed"
