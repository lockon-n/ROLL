#!/bin/bash

MODEL=${1:?"Usage: $0 <model_name> [port]"}
PORT=${2:-8000}

curl -N "http://localhost:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello, who are you?\"}],\"max_tokens\":128,\"stream\":true}"
