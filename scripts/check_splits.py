"""Audit dataset manifests for duplicate images and cross-split leakage."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lwcamo.data import read_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+")
    args = parser.parse_args()
    memberships = {}
    for manifest in args.manifests:
        images = [image for image, _ in read_pairs(manifest)]
        if len(images) != len(set(images)):
            raise ValueError(f"Duplicate images within {manifest}")
        for image in images:
            memberships.setdefault(image, []).append(manifest)
    leakage = {image: locations for image, locations in memberships.items() if len(locations) > 1}
    if leakage:
        preview = list(leakage.items())[:10]
        raise ValueError(f"Cross-manifest leakage ({len(leakage)} images): {preview}")
    print(f"Split audit: PASS ({len(memberships)} unique images)")


if __name__ == "__main__":
    main()
