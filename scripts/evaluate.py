"""Compute S-alpha, adaptive E-measure, weighted F-measure, and MAE."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lwcamo.evaluation import evaluate_directories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate_directories(args.predictions, args.masks)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
