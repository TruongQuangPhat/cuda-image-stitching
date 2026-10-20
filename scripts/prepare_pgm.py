#!/usr/bin/env python3
"""
Prepares grayscale PGM (P5) image files from prepared JPEG images for S1-S4
integration tests in cuda-image-stitching.

Reads dataset/metadata.csv, locates prepared JPEG images, and converts them
to P5 PGM format without modifying or deleting the original JPEGs.

Requirements:
    Python 3.8+
    Pillow (PIL): pip install Pillow
"""

import csv
import os
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Install via: pip install Pillow", file=sys.stderr)
    sys.exit(1)


def convert_jpg_to_pgm(src_path: Path, dst_path: Path) -> bool:
    if not src_path.is_file():
        return False
    try:
        with Image.open(src_path) as im:
            gray = im.convert("L")
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            gray.save(dst_path, format="PPM")  # PIL uses PPM format for PGM when mode is 'L'
        return True
    except Exception as e:
        print(f"Error converting {src_path} -> {dst_path}: {e}", file=sys.stderr)
        return False


def main():
    repo_root = Path(__file__).resolve().parent.parent
    metadata_path = repo_root / "dataset" / "metadata.csv"

    if not metadata_path.is_file():
        print(f"Error: metadata.csv not found at {metadata_path}", file=sys.stderr)
        sys.exit(1)

    dataset_dir = repo_root / "dataset"
    converted_count = 0
    total_images = 0

    print(f"Reading dataset configuration from {metadata_path}...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair_id = f"{row['pair_id']}_{row['resolution']}"
            rel_a = row["prepared_a"]
            rel_b = row["prepared_b"]

            src_a = dataset_dir / rel_a
            src_b = dataset_dir / rel_b

            dst_a = src_a.with_suffix(".pgm")
            dst_b = src_b.with_suffix(".pgm")

            total_images += 2

            ok_a = convert_jpg_to_pgm(src_a, dst_a)
            ok_b = convert_jpg_to_pgm(src_b, dst_b)

            if ok_a and ok_b:
                converted_count += 2
                print(f"  [OK] {pair_id}: {dst_a.relative_to(repo_root)} and {dst_b.relative_to(repo_root)}")
            else:
                print(f"  [FAIL] {pair_id}: Failed to convert one or both images")

    print(f"\nCompleted: {converted_count}/{total_images} images prepared as PGM.")
    if converted_count < total_images:
        sys.exit(1)


if __name__ == "__main__":
    main()
