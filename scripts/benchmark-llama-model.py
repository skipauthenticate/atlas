#!/usr/bin/env python3
"""Run one audited, pinned llama-bench candidate from a source checkout.

Example:
  python scripts/benchmark-llama-model.py \
    --profile light \
    --inventory benchmarks/voice_models_q8_inventory.json \
    --llama-bench /absolute/path/to/llama-bench \
    --output data/artifacts/voice-model-benchmark/llama-light.json

The output path must be a new, non-symlinked ``.json`` file. The command uses a
fixed 512-token prompt, 128-token generation, five repetitions, full GPU
offload, flash attention, and Q8_0 KV caches. Threads use the binary's platform
default and the observed value is recorded in the output.
"""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from atlas_voice.llama_bench_wrapper import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
