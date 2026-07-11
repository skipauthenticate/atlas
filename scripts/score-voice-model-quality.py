#!/usr/bin/env python3
"""Score a completed voice-model benchmark against the versioned quality corpus."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from atlas_voice.quality_postprocessor import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
