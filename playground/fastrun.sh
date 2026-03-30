#!/bin/bash

MODEL=${1:?"Usage: $0 <model_name> [port]"}
PORT=${2:-8000}

while true; do

curl -sN "http://localhost:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello, who are you?\"}],\"max_tokens\":128,\"stream\":true}" \
  | sed -u 's/^data: //;/^\[DONE\]/d;/^$/d' \
  | while IFS= read -r line; do
      printf '%s' "$(echo "$line" | python3 -c "import sys,json; c=json.load(sys.stdin)['choices'][0]['delta'].get('content',''); print(c,end='')")"
    done
echo

done