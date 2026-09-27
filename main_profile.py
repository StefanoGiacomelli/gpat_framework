"""Root orchestrator for unified GP-AT model profiling.

Results are written as one JSON file per model profiling execution under
``profile_results/`` by default, using ``<model_name>_<timestamp>.json``.
"""

from __future__ import annotations

from inference.profiling.profile_main import main


if __name__ == "__main__":
    raise SystemExit(main())

