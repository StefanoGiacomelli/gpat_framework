"""Root entry point for the GP-AT real-time inference GUI."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REAL_TIME_ROOT = ROOT / "inference" / "real_time"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REAL_TIME_ROOT) not in sys.path:
    sys.path.insert(0, str(REAL_TIME_ROOT))

from inference.real_time.inference_gui import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
