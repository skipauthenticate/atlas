#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

failed=false

report_matches() {
  local label="$1"
  local matches="$2"

  if [[ -n "$matches" ]]; then
    printf '\n%s\n' "$label" >&2
    printf '%s\n' "$matches" >&2
    failed=true
  fi
}

check_pattern() {
  local label="$1"
  local pattern="$2"
  local matches

  matches="$(
    git grep -n -I -E "$pattern" -- . \
      ':(exclude)scripts/check-open-source-ready.sh' \
      || true
  )"
  report_matches "$label" "$matches"
}

check_pattern \
  "Found host-specific paths. Use relative paths or placeholders instead:" \
  '(/home/[A-Za-z0-9._-]+/|/Users/[A-Za-z0-9._-]+/|[A-Za-z]:\\Users\\|workspace/jetson-qwen|qwen3\.6-35b-a3b-mtp|Qwen3\.6-35B-A3B-MXFP4_MOE\.gguf)'

check_pattern \
  "Found credential-like material. Keep real secrets out of git:" \
  '(github_pat_[A-Za-z0-9_]+|ghp_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|BEGIN (RSA |OPENSSH |EC |DSA )?PRIVATE KEY)'

sensitive_paths=()
while IFS= read -r -d '' path; do
  case "$path" in
    .env|.env.*)
      if [[ "$path" != ".env.example" ]]; then
        sensitive_paths+=("$path")
      fi
      ;;
    data/*|models/*|cache/*|logs/*|.run/*)
      sensitive_paths+=("$path")
      ;;
    *.gguf|*.safetensors|*.pt|*.pth|*.ckpt|*.onnx)
      sensitive_paths+=("$path")
      ;;
    *.wav|*.mp3|*.m4a|*.flac)
      sensitive_paths+=("$path")
      ;;
    *.sqlite|*.sqlite-shm|*.sqlite-wal|*.log)
      sensitive_paths+=("$path")
      ;;
  esac
done < <(git ls-files -z)

if (( ${#sensitive_paths[@]} > 0 )); then
  {
    printf '\nFound tracked local/private artifacts. Remove these from git:\n'
    printf '%s\n' "${sensitive_paths[@]}"
  } >&2
  failed=true
fi

if [[ "$failed" == true ]]; then
  exit 1
fi

printf 'Open-source readiness checks passed.\n'
