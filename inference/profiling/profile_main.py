"""CLI wrapper for the unified GP-AT profiling API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.profiling.profile_utils import (  # noqa: E402
    DEFAULT_DURATION_SECONDS,
    DEFAULT_MEASURED_RUNS,
    DEFAULT_WARMUP_RUNS,
    ModelProfiler,
    ModelRegistry,
    ResultsManager,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile local GP-AT models.")
    parser.add_argument("--models-dir", default="models", help="Path to the local models directory.")
    parser.add_argument("--output-dir", default="profile_results", help="Directory for timestamped JSON results.")
    parser.add_argument("--models", nargs="+", help="Optional list of model names to profile.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Requested execution device.")
    parser.add_argument("--duration-seconds", type=float, default=DEFAULT_DURATION_SECONDS, help="Dummy input duration.")
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS, help="Warm-up runs excluded from timing.")
    parser.add_argument("--measured-runs", type=int, default=DEFAULT_MEASURED_RUNS, help="Measured inference runs.")
    parser.add_argument("--skip-complexity", action="store_true", help="Skip FLOPs/MACs and torchinfo analysis.")
    parser.add_argument("--skip-minimum-input", action="store_true", help="Skip minimum accepted input search.")
    
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = ModelRegistry(args.models_dir)
    specs = registry.discover(args.models)
    if not specs:
        requested = ", ".join(args.models or ["all known models"])
        print(f"No model specs with local checkpoints found for: {requested}", file=sys.stderr)
        return 1

    profiler = ModelProfiler(models_dir=args.models_dir,
                             requested_device=args.device,
                             duration_seconds=args.duration_seconds,
                             warmup_runs=args.warmup_runs,
                             measured_runs=args.measured_runs,
                             run_complexity=not args.skip_complexity,
                             run_minimum_input=not args.skip_minimum_input)
    results = ResultsManager(args.output_dir)

    failures = 0
    for spec in specs:
        print(f"Profiling {spec.name} on requested device '{args.device}'...")
        result = profiler.profile(spec)
        output_path = results.save(result)
        status = result.get("status")
        if status != "success":
            failures += 1
        effective = result.get("environment", {}).get("effective_device")
        fallback = result.get("environment", {}).get("fallback_reason")
        suffix = f" fallback={fallback}" if fallback else ""
        print(f"  {status}: saved {output_path} effective_device={effective}{suffix}")

    print(f"Completed {len(specs)} model profile attempt(s); failures={failures}.")
    
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
