#!/usr/bin/env bash
# Start the LeafCheck Expert server on Atlas.
# Phones on the same Wi-Fi open:  http://<atlas-ip>:8080
set -e
cd "$(dirname "$0")"

[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt

export BACKEND="${BACKEND:-ollama}"        # ollama | openai | mock
export MODEL="${MODEL:-qwen3-vl:30b}"
# export ACCESS_CODE="pick-a-code"         # uncomment to require a code on phones

echo "Backend: $BACKEND   Model: $MODEL"
echo "Open on your phone: http://$(hostname -I | awk '{print $1}'):8080"
exec uvicorn app:app --host 0.0.0.0 --port 8080
