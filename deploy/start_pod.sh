#!/usr/bin/env bash
# Start ComfyUI (private, 127.0.0.1) and the Hawk H3 API gateway (public) on a GPU pod.
#
#   export HAWK_API_TOKEN=$(openssl rand -hex 24)
#   export ATLAS_API_KEY=...                       # for the story planner
#   export PUBLIC_BASE_URL=https://<pod-id>-8000.proxy.runpod.net
#   bash deploy/start_pod.sh
#
# Expose ONLY the API port (8000) on the pod. ComfyUI has no authentication.
set -euo pipefail

: "${HAWK_API_TOKEN:?Set HAWK_API_TOKEN, e.g. export HAWK_API_TOKEN=\$(openssl rand -hex 24)}"
: "${ATLAS_API_KEY:?Set ATLAS_API_KEY so the story planner can call Atlas Cloud}"

COMFY_DIR="${COMFY_DIR:-/workspace/ComfyUI}"
PACK_DIR="${PACK_DIR:-$COMFY_DIR/custom_nodes/Hawk-Minimax-H3-Directory}"
PYTHON="${PYTHON:-python}"
API_PORT="${API_PORT:-8000}"
COMFY_PORT="${COMFY_PORT:-8188}"

export COMFY_URL="http://127.0.0.1:${COMFY_PORT}"
export DATA_DIR="${DATA_DIR:-/workspace/hawk_api_data}"
export PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-http://127.0.0.1:${API_PORT}}"
export MAX_UPLOAD_MB="${MAX_UPLOAD_MB:-2048}"
mkdir -p "$DATA_DIR"

"$PYTHON" -m pip install -q -r "$PACK_DIR/requirements-api.txt"

echo "Starting ComfyUI on 127.0.0.1:${COMFY_PORT} (log: $DATA_DIR/comfyui.log)"
(cd "$COMFY_DIR" && exec "$PYTHON" main.py --listen 127.0.0.1 --port "$COMFY_PORT" \
    --max-upload-size "$MAX_UPLOAD_MB") >"$DATA_DIR/comfyui.log" 2>&1 &
COMFY_PID=$!
trap 'kill "$COMFY_PID" 2>/dev/null || true' EXIT INT TERM

for _ in $(seq 1 600); do
  if curl -fs "$COMFY_URL/queue" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$COMFY_PID" 2>/dev/null; then
    echo "ComfyUI exited during startup; see $DATA_DIR/comfyui.log" >&2
    exit 1
  fi
  sleep 2
done
curl -fs "$COMFY_URL/queue" >/dev/null || { echo "ComfyUI did not come up" >&2; exit 1; }

echo "Starting Hawk H3 API on 0.0.0.0:${API_PORT} -> ${PUBLIC_BASE_URL}"
cd "$PACK_DIR"
"$PYTHON" -m uvicorn --factory hawk_api.app:create_app --host 0.0.0.0 --port "$API_PORT" --proxy-headers
