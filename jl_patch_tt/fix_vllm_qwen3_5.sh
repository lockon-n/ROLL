#!/bin/bash
# Fix vLLM Qwen3.5 config bug: ignore_keys_at_rope_validation should be a set, not a list.
# Affects vLLM 0.17.0 and 0.18.0.

VLLM_DIR=$(python3 -c "import os, vllm; print(os.path.dirname(vllm.__file__))")
TARGET="$VLLM_DIR/transformers_utils/configs/qwen3_5.py"

if [ ! -f "$TARGET" ]; then
    echo "File not found: $TARGET"
    exit 1
fi

if grep -q 'kwargs\["ignore_keys_at_rope_validation"\] = \[' "$TARGET"; then
    sed -i 's/kwargs\["ignore_keys_at_rope_validation"\] = \[/kwargs["ignore_keys_at_rope_validation"] = {/' "$TARGET"
    sed -i '/kwargs\["ignore_keys_at_rope_validation"\] = {/{n;n;s/\]/}/}' "$TARGET"
    echo "Fixed: $TARGET"
    sed -n '/ignore_keys_at_rope_validation/,+3p' "$TARGET"
else
    echo "Already fixed or pattern not found in: $TARGET"
fi
