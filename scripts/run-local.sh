#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
  echo "Create the virtual environment and install the project first; see README.md."
  exit 1
fi
if ! command -v ollama >/dev/null; then
  echo "Install Ollama first: brew install ollama"
  exit 1
fi
mkdir -p artifacts
owned_ollama_pid=''
cleanup() {
  if [[ -n "$owned_ollama_pid" ]]; then kill "$owned_ollama_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  OLLAMA_NO_CLOUD=1 OLLAMA_HOST=127.0.0.1:11434 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=2 OLLAMA_CONTEXT_LENGTH=4096 ollama serve > artifacts/ollama.log 2>&1 &
  owned_ollama_pid=$!
  ready=0
  for _ in {1..30}; do
    if curl -fsS --max-time 1 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then ready=1; break; fi
    sleep 1
  done
  if [[ "$ready" == 0 ]]; then echo "Ollama did not start; see artifacts/ollama.log"; exit 1; fi
fi
if ! ollama show "${OLLAMA_MODEL:-llama3.2:3b}" >/dev/null 2>&1; then
  echo "Download the local model first: ollama pull ${OLLAMA_MODEL:-llama3.2:3b}"
  exit 1
fi
PORT="${PORT:-5001}" .venv/bin/python -m workflows.app
