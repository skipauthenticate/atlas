#!/usr/bin/env python3
"""Install the pinned Q8 voice-model candidates from a source checkout."""

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from atlas_voice.model_installer import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
