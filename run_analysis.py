#!/usr/bin/env python3
"""Command-line launcher for the complete ECG analysis."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent
PIPELINE = REPOSITORY_ROOT / "analysis" / "full_analysis.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ECG transition and classification analyses."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing one MATLAB .mat ECG file per participant.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "analysis_outputs",
        help="Destination for generated tables, figures and audit files.",
    )
    parser.add_argument(
        "--offsets-json",
        type=Path,
        help="Optional JSON mapping participant IDs to timeline offsets in seconds.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced smoke test; its results are not manuscript results.",
    )
    return parser.parse_args()


def require_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise SystemExit(f"Data directory does not exist: {resolved}")
    if not any(resolved.glob("*.mat")):
        raise SystemExit(f"No .mat files found in: {resolved}")
    return resolved


def main() -> int:
    args = parse_args()
    data_dir = require_directory(args.data_dir)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment["ANXIETY_ECG_DATA_DIR"] = str(data_dir)
    environment["ANXIETY_ECG_OUTPUT_DIR"] = str(output_dir)
    environment.setdefault("MPLBACKEND", "Agg")

    if args.offsets_json:
        offsets_path = args.offsets_json.expanduser().resolve()
        if not offsets_path.is_file():
            raise SystemExit(f"Offsets file does not exist: {offsets_path}")
        environment["ANXIETY_ECG_OFFSETS_JSON"] = str(offsets_path)

    if args.quick:
        environment["ANXIETY_ECG_RUN_TSFRESH"] = "0"
        environment["ANXIETY_ECG_RUN_SECONDARY_MODELS"] = "0"
        environment["ANXIETY_ECG_N_SIGNFLIP"] = "500"
        environment["ANXIETY_ECG_N_BOOT"] = "200"
        environment["ANXIETY_ECG_N_TRANSITION_PERM"] = "50"

    completed = subprocess.run(
        [sys.executable, str(PIPELINE)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
