"""Train LWCamo-S/M/L with the same leakage-free CAMO+COD10K protocol."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TRAIN_MANIFEST = ROOT / "datasets" / "splits" / "train_camo_cod10k.txt"
VALID_MANIFEST = ROOT / "datasets" / "splits" / "valid_camo_cod10k.txt"


def main():
    for variant in ("small", "medium", "large"):
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--config",
            str(ROOT / "configs" / f"lwcamo_{variant}.json"),
            "--train-list",
            str(TRAIN_MANIFEST),
            "--valid-list",
            str(VALID_MANIFEST),
            "--output-dir",
            str(ROOT / "results" / f"lwcamo_{variant}"),
        ]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
