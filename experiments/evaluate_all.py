"""Evaluate the standard four COD test sets."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
for dataset in ("CAMO", "COD10K", "CHAMELEON", "NC4K"):
    command = [sys.executable, str(ROOT / "scripts" / "evaluate.py"),
               "--predictions", str(ROOT / "results" / "predictions" / dataset),
               "--masks", str(ROOT / "datasets" / dataset / "GT"),
               "--output", str(ROOT / "results" / "metrics" / f"{dataset.lower()}.json")]
    if subprocess.call(command):
        raise SystemExit(f"Evaluation failed for {dataset}")
