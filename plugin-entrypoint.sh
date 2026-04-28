#!/usr/bin/env sh
set -eu

# Enable pywaggle publish path by default inside plugin runtime.
export PYWAGGLE_PUBLISH_ENABLED="${PYWAGGLE_PUBLISH_ENABLED:-true}"

# Keep defaults local to the container unless overridden by plugin args/env.
export GATEWAY_HOST="${GATEWAY_HOST:-0.0.0.0}"
export GATEWAY_PORT="${GATEWAY_PORT:-8080}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000}"
export TORCH_WORKER_URL="${TORCH_WORKER_URL:-http://127.0.0.1:8001}"

exec sage-gateway
