#!/bin/bash
VLLM_DIR=$(python3 -c "import os, vllm; print(os.path.dirname(vllm.__file__))")
export TARGET="$VLLM_DIR/transformers_utils/configs/qwen3_5.py"

if [ ! -f "$TARGET" ]; then
    echo "File not found: $TARGET"
    exit 1
fi

python3 << 'EOF'
import re, sys, os

target = os.environ["TARGET"]

with open(target) as f:
    text = f.read()

pattern = r'(kwargs\["ignore_keys_at_rope_validation"\]\s*=\s*)\[(.*?)\]'
new_text = re.sub(pattern, r'\1{\2}', text, count=1, flags=re.DOTALL)

if new_text == text:
    print(f"Already fixed or pattern not found in: {target}")
else:
    with open(target, "w") as f:
        f.write(new_text)
    print(f"Fixed: {target}")
EOF
