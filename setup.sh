#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

SMOKE_TEST=false
NO_START=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke-test)
      SMOKE_TEST=true
      shift
      ;;
    --no-start)
      NO_START=true
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

log() {
  printf '[atlas-voice] %s\n' "$*"
}

is_jetson() {
  [[ -f /etc/nv_tegra_release ]] || grep -qi 'tegra' /proc/device-tree/model 2>/dev/null
}

compose() {
  if docker info >/dev/null 2>&1; then
    docker compose "$@"
  else
    sudo docker compose "$@"
  fi
}

install_host_packages() {
  if ! command -v apt-get >/dev/null 2>&1; then
    log "apt-get not found; skipping host package installation"
    return
  fi

  local packages=(docker.io docker-compose-plugin ffmpeg git-lfs curl)
  log "Installing host packages with sudo: ${packages[*]}"
  sudo apt-get update
  sudo apt-get install -y "${packages[@]}"

  if apt-cache show nvidia-container-toolkit >/dev/null 2>&1; then
    log "Installing NVIDIA container toolkit"
    sudo apt-get install -y nvidia-container-toolkit
  else
    log "nvidia-container-toolkit is not available in apt sources; skipping toolkit install"
  fi

  if command -v nvidia-ctk >/dev/null 2>&1; then
    log "Configuring NVIDIA container runtime"
    sudo nvidia-ctk runtime configure --runtime=docker || true
    sudo systemctl restart docker || true
  fi
}

ensure_env() {
  if [[ ! -f .env ]]; then
    cp .env.example .env
    log "Created .env from .env.example"
  fi
}

get_env() {
  local key="$1"
  grep -E "^${key}=" .env | tail -n1 | cut -d= -f2- || true
}

set_env() {
  local key="$1"
  local value="$2"
  local escaped
  escaped="$(printf '%s' "$value" | sed 's/[\/&]/\\&/g')"
  if grep -qE "^${key}=" .env; then
    sed -i "s/^${key}=.*/${key}=${escaped}/" .env
  else
    printf '%s=%s\n' "$key" "$value" >> .env
  fi
}

prompt_hf_token() {
  local token
  token="$(get_env HF_TOKEN)"
  if [[ -n "$token" ]]; then
    return
  fi
  printf 'HF_TOKEN for pyannote gated model access: '
  read -rs token
  printf '\n'
  if [[ -z "$token" ]]; then
    log "HF_TOKEN is empty; setup can continue, but diarization will fail until it is set"
    return
  fi
  set_env HF_TOKEN "$token"
}

verify_pyannote_access() {
  local token
  token="$(get_env HF_TOKEN)"
  if [[ -z "$token" ]]; then
    return
  fi

  log "Checking Hugging Face access for pyannote community diarization"
  local code
  code="$(curl -sS -o /tmp/atlas_voice_hf_check.json -w '%{http_code}' \
    -H "Authorization: Bearer ${token}" \
    https://huggingface.co/api/models/pyannote/speaker-diarization-community-1 || true)"
  if [[ "$code" != "200" ]]; then
    cat <<'MSG'
Hugging Face did not allow access to pyannote/speaker-diarization-community-1.
Confirm the token is valid and accept the model terms in your browser:

  https://huggingface.co/pyannote/speaker-diarization-community-1

Then rerun ./setup.sh.
MSG
  fi
}

prepare_llm_model() {
  mkdir -p models/llm
  local target="models/llm/model.gguf"
  local target_abs="${ROOT_DIR}/${target}"
  if [[ -e "$target" ]]; then
    set_env LLM_MODEL_PATH "$target_abs"
    log "LLM model already present at $target"
    return
  fi

  local configured
  configured="$(get_env LLM_MODEL_PATH)"
  if [[ -n "$configured" && -f "$configured" ]]; then
    set_env LLM_MODEL_PATH "$configured"
    log "Using LLM model from $configured"
    return
  fi

  local default_path="/home/atlas/workspace/jetson-qwen/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-MXFP4_MOE.gguf"
  if [[ -f "$default_path" ]]; then
    set_env LLM_MODEL_PATH "$default_path"
    log "Using LLM model from $default_path"
    return
  fi

  local model_url
  model_url="$(get_env LLM_MODEL_URL)"
  if [[ -n "$model_url" ]]; then
    log "Downloading LLM model from LLM_MODEL_URL"
    curl -L "$model_url" -o "$target"
    set_env LLM_MODEL_PATH "$target_abs"
    return
  fi

  printf 'Path to local Qwen GGUF model, or leave blank to skip for now: '
  read -r configured
  if [[ -n "$configured" && -f "$configured" ]]; then
    set_env LLM_MODEL_PATH "$configured"
    log "Using LLM model from $configured"
  else
    set_env LLM_MODEL_PATH "$target_abs"
    log "No LLM model configured; llama.cpp container will not start successfully until models/llm/model.gguf exists"
  fi
}

prepare_dirs() {
  mkdir -p data/inbox data/originals data/normalized data/artifacts
  mkdir -p models/asr models/diarization models/llm cache/huggingface
}

smoke_test() {
  log "Creating smoke-test audio"
  ffmpeg -hide_banner -loglevel error -y \
    -f lavfi -i sine=frequency=440:duration=15 \
    -ar 16000 -ac 1 data/inbox/smoke-test.wav
  log "Queued smoke-test audio through inbox. Watch progress with: atlas-voice logs"
}

main() {
  if is_jetson; then
    log "Jetson/L4T host detected"
  else
    log "Jetson/L4T host not detected; continuing with generic Linux defaults"
  fi

  install_host_packages
  ensure_env
  prompt_hf_token
  verify_pyannote_access
  prepare_dirs
  prepare_llm_model

  log "Building containers"
  compose build

  if [[ "$NO_START" == false ]]; then
    log "Starting Atlas Voice"
    compose up -d
    if [[ "$SMOKE_TEST" == true ]]; then
      smoke_test
    fi
    log "Open http://127.0.0.1:$(get_env ATLAS_VOICE_PORT)"
  fi
}

main
